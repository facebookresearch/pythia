# Copyright (c) Facebook, Inc. and its affiliates.

# Mostly copy-pasted from
# https://github.com/facebookresearch/detr/blob/master/models/matcher.py
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from torch import nn

from scipy.optimize import linear_sum_assignment

from ..util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
        logsoftmax: bool = False,
    ):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.norm = nn.LogSoftmax(-1) if logsoftmax else nn.Softmax(-1)
        assert (
            cost_class != 0 or cost_bbox != 0 or cost_giou != 0
        ), "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = self.norm(
            outputs["pred_logits"].flatten(0, 1)
        )  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox).float(), box_cxcywh_to_xyxy(tgt_bbox)
        )

        # Final cost matrix
        C = (
            self.cost_bbox * cost_bbox
            + self.cost_class * cost_class
            + self.cost_giou * cost_giou
        )
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [
            linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))
        ]
        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]


class SequentialMatcher(nn.Module):
    def forward(self, outputs, targets):
        return [(torch.arange(len(tgt["labels"])),) * 2 for tgt in targets]


class LexicographicalMatcher(nn.Module):
    def __init__(self, lexic="acb"):
        super().__init__()
        self.lexic = lexic

    def forward(self, outputs, targets):
        indices = []
        for tgt in targets:
            tgt_cls, tgt_box = tgt["labels"], tgt["boxes"]
            area = tgt_box[:, 2] * tgt_box[:, 3]
            if self.lexic == "acb":
                search_list = [
                    (-a, cl, b)
                    for cl, a, b in zip(
                        tgt_cls.tolist(), area.tolist(), tgt_box.tolist()
                    )
                ]
            else:
                search_list = [
                    (cl, -a, b)
                    for cl, a, b in zip(
                        tgt_cls.tolist(), area.tolist(), tgt_box.tolist()
                    )
                ]
            # argsort from https://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python
            j = sorted(range(len(search_list)), key=search_list.__getitem__)
            j = torch.as_tensor(j, dtype=torch.int64)
            i = torch.arange(len(j), dtype=j.dtype)
            indices.append((i, j))
        return indices


def build_matcher(args):
    if args.set_loss == "sequential":
        matcher = SequentialMatcher()
    elif args.set_loss == "hungarian":
        matcher = HungarianMatcher(
            cost_class=args.set_cost_class,
            cost_bbox=args.set_cost_bbox,
            cost_giou=args.set_cost_giou,
            logsoftmax=args.use_bcl,
        )
    elif args.set_loss == "lexicographical":
        matcher = LexicographicalMatcher()
    else:
        raise ValueError(
            f"Only sequential, lexicographical and hungarian accepted, got {args.set_loss}"
        )
    return matcher
