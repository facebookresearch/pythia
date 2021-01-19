# Copyright (c) Facebook, Inc. and its affiliates.

# Mostly copy-pasted from
# https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from ..util import box_ops
from ..util.misc import (
    NestedTensor,
    accuracy,
    get_world_size,
    is_dist_avail_and_initialized,
)
from .backbone import build_backbone
from .matcher import build_matcher
from .transformer import build_transformer


ATTRIBUTE_CLASS_NUM = 401
MAX_ATTR_NUM = 16


class AttributeHead(nn.Module):
    def __init__(self, num_classes, representation_size):
        super().__init__()

        # copy-pasted from
        # https://github.com/ronghanghu/vqa-maskrcnn-benchmark-m4c/blob/0fbccee2dfed10d652bcf014f9f8bfafd8478f51/maskrcnn_benchmark/modeling/roi_heads/box_head/roi_box_predictors.py#L52-L61
        self.cls_embed = nn.Embedding(num_classes + 1, 256)
        self.attr_linear1 = nn.Linear(representation_size + 256, 512)
        self.attr_linear2 = nn.Linear(512, ATTRIBUTE_CLASS_NUM)

        nn.init.normal_(self.cls_embed.weight, std=0.01)
        nn.init.normal_(self.attr_linear1.weight, std=0.01)
        nn.init.normal_(self.attr_linear2.weight, std=0.01)
        nn.init.constant_(self.attr_linear1.bias, 0)
        nn.init.constant_(self.attr_linear2.bias, 0)

    def forward(self, hidden_states, labels):
        # copy-pasted from
        # https://github.com/ronghanghu/vqa-maskrcnn-benchmark-m4c/blob/0fbccee2dfed10d652bcf014f9f8bfafd8478f51/maskrcnn_benchmark/modeling/roi_heads/box_head/roi_box_predictors.py#L76-L96

        # get embeddings of indices using gt cls labels
        cls_embed_out = self.cls_embed(labels)

        # concat with fc7 feats
        concat_attr = torch.cat([hidden_states, cls_embed_out], dim=-1)

        # pass through attr head layers
        fc_attr = self.attr_linear1(concat_attr)
        attr_score = F.relu(self.attr_linear2(fc_attr))

        return attr_score


class DETR(nn.Module):
    """ This is the DETR module that performs object detection """

    def __init__(self, backbone, transformer, num_queries):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        encoder_hidden_dim = transformer.d_model_enc
        decoder_hidden_dim = transformer.d_model_dec

        self.query_embeds = nn.ModuleDict()
        for task_type in self.num_queries:
            task_dict = nn.ModuleDict()
            for dataset in self.num_queries[task_type]:
                task_dict[dataset] = nn.Embedding(
                    self.num_queries[task_type][dataset], decoder_hidden_dim
                )
            self.query_embeds[task_type] = task_dict

        self.input_proj = nn.Conv2d(
            backbone.num_channels, encoder_hidden_dim, kernel_size=1
        )
        self.backbone = backbone

    def forward(
        self,
        img_src: NestedTensor = None,
        text_src=None,
        text_mask=None,
        text_pos=None,
        output_hidden_states_only=False,
        task_type="detection",
        dataset_name="detection_coco",
        task_idx=None,
        **kwargs,
    ):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        img_mask = None
        img_pos = [None]
        if img_src is not None:
            if not isinstance(img_src, NestedTensor):
                img_src = NestedTensor.from_tensor_list(img_src)
            features, img_pos = self.backbone(img_src)

            img_src, img_mask = features[-1].decompose()
            img_src = self.input_proj(img_src)

        query_embed = self.query_embeds[task_type][dataset_name]
        hs, _ = self.transformer(
            img_src=img_src,
            img_mask=img_mask,
            img_pos=img_pos[-1],
            text_src=text_src,
            text_mask=text_mask,
            text_pos=text_pos,
            query_embed=query_embed.weight,
            task_type=task_type,
            dataset_name=dataset_name,
            task_idx=task_idx,
        )

        if hs is not None:
            assert hs.size(2) == self.num_queries[task_type][dataset_name]

        return {"hidden_states": hs}


def _attribute_loss(attribute_logits, attributes):
    _N, _B, _C = attribute_logits.size()
    assert _C == ATTRIBUTE_CLASS_NUM
    attribute_logits = attribute_logits.view(_N * _B, _C)

    assert attributes.size(0) == _N
    assert attributes.size(1) == _B
    assert attributes.size(2) == MAX_ATTR_NUM
    attributes = attributes.view(_N * _B, MAX_ATTR_NUM)

    # https://github.com/ronghanghu/vqa-maskrcnn-benchmark-m4c/blob/0fbccee2dfed10d652bcf014f9f8bfafd8478f51/maskrcnn_benchmark/modeling/roi_heads/box_head/loss.py#L185-L222
    n_boxes = attribute_logits.shape[0]

    # N_BOXES, N_ATTR -> N_BOXES, 1, N_ATTR
    attribute_logits = attribute_logits.unsqueeze(1)

    # N_BOXES, 1, N_ATTR -> N_BOXES, MAX_ATTR_PER_INST, N_ATTR -> N_BOXES * MAX_ATTR_PER_INST, N_ATTR
    attribute_logits = (
        attribute_logits.expand(n_boxes, 16, ATTRIBUTE_CLASS_NUM)
        .contiguous()
        .view(-1, ATTRIBUTE_CLASS_NUM)
    )

    # Normalize each box loss by # of valid GT attributes (ie attr != -1)
    # Repeat number of valid attributes per box along the rows and take transpose
    inv_per_box_weights = (
        (attributes >= 0).sum(dim=1).repeat(16, 1).transpose(0, 1).flatten()
    )
    per_box_weights = inv_per_box_weights.float().reciprocal()
    per_box_weights[per_box_weights > 1] = 0.0

    attributes = attributes.view(-1)
    attribute_loss = 0.5 * F.cross_entropy(
        attribute_logits, attributes, reduction="none", ignore_index=-1
    )

    attribute_loss = (attribute_loss * per_box_weights).view(n_boxes, -1).sum(dim=1)

    # Find number of boxes with atleast valid attribute
    n_valid_boxes = len(attribute_loss.nonzero())

    if n_valid_boxes > 0:
        attribute_loss = (attribute_loss / n_valid_boxes).sum()
    else:
        attribute_loss = (attribute_loss * 0.0).sum()

    return attribute_loss


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(
        self, num_classes, matcher, weight_dict, eos_coef, losses, attribute_head=None
    ):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)
        self.attribute_head = attribute_head

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2), target_classes, self.empty_weight
        )
        losses = {"loss_ce": loss_ce}

        if self.attribute_head is not None and "attributes" in targets[0]:
            attribute_logits = self.attribute_head(
                outputs["hs_for_attr"], target_classes
            )
            target_attributes_o = torch.cat(
                [t["attributes"][J] for t, (_, J) in zip(targets, indices)]
            )
            target_attributes = -torch.ones(
                *src_logits.shape[:2], 16, dtype=torch.int64, device=src_logits.device
            )
            target_attributes[idx] = target_attributes_o
            losses["loss_attr"] = _attribute_loss(attribute_logits, target_attributes)

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses["class_error"] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_labels_balanced(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        sl = src_logits.flatten(0, 1)
        tc = target_classes.flatten(0, 1)
        pos = tc != self.num_classes
        loss_pos = F.cross_entropy(sl[pos], tc[pos], reduction="none").sum() / num_boxes
        loss_neg = F.cross_entropy(sl[~pos], tc[~pos], reduction="none").sum() / (
            sl.shape[0] - num_boxes
        )

        loss_ce = (1 - self.eos_coef) * loss_pos + self.eos_coef * loss_neg
        losses = {"loss_ce": loss_ce}

        if self.attribute_head is not None:
            raise NotImplementedError()

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses["class_error"] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs["pred_logits"]
        device = pred_logits.device
        tgt_lengths = torch.as_tensor(
            [len(v["labels"]) for v in targets], device=device
        )
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {"cardinality_error": card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes).float(),
                box_ops.box_cxcywh_to_xyxy(target_boxes),
            )
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        from ..util.misc import interpolate
        from .segmentation import dice_loss, sigmoid_focal_loss

        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = NestedTensor.from_tensor_list(
            [t["masks"] for t in targets]
        ).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(
            src_masks[:, None],
            size=target_masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "labels_balanced": self.loss_labels_balanced,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "masks": self.loss_masks,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == "masks":
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss in ("labels", "labels_balanced"):
                        # Logging is enabled only for the last layer
                        kwargs = {"log": False}
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices, num_boxes, **kwargs
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    # num_classes = 20 if args.dataset_file != 'coco' else 91
    # if args.dataset_file == "lvis":
    #     num_classes = 1235
    # if args.dataset_file == "coco_panoptic":
    #     num_classes = 250
    # if args.dataset_file == "visualgenome":
    #     num_classes = 1601
    assert not args.masks or args.mask_model != "none"

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = DETR(backbone, transformer, num_queries=args.num_queries)
    if args.mask_model != "none":
        from .segmentation import DETRsegm

        model = DETRsegm(
            model,
            mask_head=args.mask_model,
            freeze_detr=(args.frozen_weights is not None),
        )

    return model


def build_detection_loss(args, num_classes, attribute_head):
    matcher = build_matcher(args)
    weight_dict = {"loss_ce": 1, "loss_bbox": args.bbox_loss_coef}
    weight_dict["loss_giou"] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    weight_dict["loss_attr"] = args.attr_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = []
    if args.use_bcl:
        losses.append("labels_balanced")
    else:
        losses.append("labels")
    losses.extend(["boxes", "cardinality"])
    if args.masks:
        losses += ["masks"]

    criterion = SetCriterion(
        num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=args.eos_coef,
        losses=losses,
        attribute_head=attribute_head,
    )

    return criterion
