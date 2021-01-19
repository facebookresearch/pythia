# Copyright (c) Facebook, Inc. and its affiliates.
# Initial version was taken from https://github.com/uclanlp/visualbert
# which was cleaned up and adapted for MMF.

import logging
from copy import deepcopy

import torch
from mmf.common.registry import registry
from mmf.models import BaseModel
from mmf.modules.detr.models.detr import (
    MLP,
    AttributeHead,
    build as build_model,
    build_detection_loss,
)
from mmf.modules.encoders import TransformerEncoder
from torch import nn
from transformers.modeling_bert import BertPredictionHeadTransform


logger = logging.getLogger(__name__)


@registry.register_model("unit")
class UniT(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

    @classmethod
    def config_path(cls):
        return "configs/models/unit/defaults.yaml"

    def build(self):
        # build the image encoder (DETR)
        self.detr_model = build_model(self.config.detr_args)

        def keep_only_backbone_params(model_state_dict):
            keys = list(model_state_dict.keys())
            for k in keys:
                if "backbone" not in k:
                    model_state_dict.pop(k)

        ckpt_path = self.config.detr_ckpt_path
        if ckpt_path != "":
            logger.info("initializing DETR from {}".format(ckpt_path))
            if ckpt_path.startswith("https"):
                detr_checkpoint = torch.hub.load_state_dict_from_url(
                    ckpt_path, check_hash=True
                )
            else:
                detr_checkpoint = torch.load(ckpt_path)
            if self.config.detr_ckpt_load_backbone_only:
                keep_only_backbone_params(detr_checkpoint["model"])
                self.detr_model.load_state_dict(detr_checkpoint["model"], strict=False)
            else:
                self.detr_model.load_state_dict(detr_checkpoint["model"], strict=True)
        else:
            logger.info("initializing DETR from scratch")

        # build the text encoder (BERT)
        self.bert_model = TransformerEncoder(self.config.bert_config)
        detr_hidden_dim = self.config.detr_args.decoder_hidden_dim
        bert_config = deepcopy(self.bert_model.config)
        self.bert_projection = nn.Linear(bert_config.hidden_size, detr_hidden_dim)
        self.bert_pos_projection = nn.Linear(bert_config.hidden_size, detr_hidden_dim)

        self.classifiers = nn.ModuleDict()

        self.task_embeddings_lang = nn.Identity()
        if self.config.detr_args.use_task_embedding_in_lang_encoder:
            self.task_embeddings_lang = nn.Embedding(
                self.config.max_task_num, bert_config.hidden_size
            )

        bert_config.hidden_size = detr_hidden_dim

        # build the task-specific output heads
        self.class_embeds = nn.ModuleDict()
        self.bbox_embeds = nn.ModuleDict()
        self.det_losses = nn.ModuleDict()
        for dataset_name in self.config.detr_args.num_queries.get("detection", []):
            num_cls = self.config.datasets["detection"][dataset_name]["num_classes"]
            self.class_embeds[dataset_name] = nn.Linear(detr_hidden_dim, num_cls + 1)
            self.bbox_embeds[dataset_name] = MLP(detr_hidden_dim, detr_hidden_dim, 4, 3)
            attr_head = None
            if self.config.datasets["detection"][dataset_name]["use_attr"]:
                attr_head = AttributeHead(num_cls, detr_hidden_dim)
            self.det_losses[dataset_name] = build_detection_loss(
                self.config.detr_args, num_cls, attr_head
            )

        vl_classifiers = nn.ModuleDict()
        for dataset_name in self.config.detr_args.num_queries.get("vl", []):
            vl_classifiers[dataset_name] = nn.Sequential(
                BertPredictionHeadTransform(bert_config),
                nn.Linear(
                    bert_config.hidden_size,
                    self.config.datasets["vl"][dataset_name]["num_labels"],
                ),
            )

        self.classifiers["vl"] = vl_classifiers
        self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)

        glue_classifiers = nn.ModuleDict()
        for dataset_name in self.config.detr_args.num_queries.get("glue", []):
            glue_classifiers[dataset_name] = nn.Sequential(
                BertPredictionHeadTransform(bert_config),
                nn.Linear(
                    bert_config.hidden_size,
                    self.config.datasets["glue"][dataset_name]["num_labels"],
                ),
            )

        self.classifiers["glue"] = glue_classifiers

        self.loss_calculation_fn = {}
        self.loss_calculation_fn["detection"] = self.detection_loss_calculation
        self.loss_calculation_fn["vl"] = self.classifier_loss_calculation
        self.loss_calculation_fn["glue"] = self.classifier_loss_calculation

        self.losses_dict = {
            "vl": {
                "vqa2": nn.BCEWithLogitsLoss(reduction="mean"),
                "hateful_memes": nn.functional.cross_entropy,
                "visual_entailment": nn.functional.cross_entropy,
            },
            "glue": {
                "glue_qnli": nn.functional.cross_entropy,
                "glue_qqp": nn.functional.cross_entropy,
                "glue_mnli_mismatched": nn.functional.cross_entropy,
                "glue_sst2": nn.functional.cross_entropy,
            },
        }

    def forward_bert_with_task_idx(self, sample_list):
        bert = self.bert_model.module
        input_ids = sample_list.input_ids
        attention_mask = sample_list.input_mask
        token_type_ids = sample_list.segment_ids
        device = input_ids.device

        position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device=device)

        input_shape = input_ids.size()

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        embedding_output = bert.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )

        start_idx = 0
        if self.config.detr_args.use_task_embedding_in_lang_encoder:
            bs = input_ids.size(0)
            task_idx = self.get_task_idx(sample_list.dataset_name)
            task_embed = self.task_embeddings_lang.weight[task_idx]
            task_embed = task_embed.unsqueeze(0).unsqueeze(0).repeat(bs, 1, 1)
            embedding_output = torch.cat([task_embed, embedding_output], dim=1)
            task_attention_mask = embedding_output.new_ones((bs, 1))
            attention_mask = torch.cat([task_attention_mask, attention_mask], dim=1)
            start_idx = 1

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None for _ in range(bert.config.num_hidden_layers)]
        encoder_outputs = bert.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
        )
        sequence_output = encoder_outputs[0][:, start_idx:, :]
        pos_embeddings = self.bert_model.embeddings.position_embeddings(position_ids)

        return sequence_output, pos_embeddings

    def forward(self, sample_list):
        detr_outputs = {}
        task_type = self.get_task_type(sample_list.dataset_name)
        text_src = None
        text_mask = None
        text_pos = None
        img_src = None
        if task_type == "vl" or task_type == "glue":
            if task_type == "vl":
                img_src = sample_list.detr_img
            text_src, pos_embeddings = self.forward_bert_with_task_idx(sample_list)
            # 768 -> 256
            text_src = self.bert_projection(text_src)
            text_pos = self.bert_pos_projection(pos_embeddings)

            text_mask = sample_list.input_mask
            if self.config.keep_only_bert_cls[task_type][sample_list.dataset_name]:
                # take only the [CLS] token's hidden state
                text_src = text_src[:, 0:1, :]
                text_pos = text_pos[0:1, :]
                text_mask = text_mask[:, 0:1]
        elif task_type == "detection":
            img_src = sample_list.detr_img

        detr_outputs = self.detr_model(
            img_src=img_src,
            text_src=text_src,
            text_mask=text_mask,
            text_pos=text_pos,
            task_type=task_type,
            dataset_name=sample_list.dataset_name,
            task_idx=self.get_task_idx(sample_list.dataset_name),
        )

        output = self.loss_calculation_fn[task_type](detr_outputs, sample_list)
        return output

    def detection_loss_calculation(self, detr_outputs, sample_list):
        hs = detr_outputs["hidden_states"]

        outputs_class = self.class_embeds[sample_list.dataset_name](hs)
        outputs_coord = self.bbox_embeds[sample_list.dataset_name](hs).sigmoid()
        detr_outputs.update(
            {
                "pred_logits": outputs_class[-1],
                "pred_boxes": outputs_coord[-1],
                "hs_for_attr": hs[-1],
            }
        )
        if self.config.detr_args.aux_loss:
            detr_outputs["aux_outputs"] = [
                {"pred_logits": a, "pred_boxes": b, "hs_for_attr": c}
                for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], hs[:-1])
            ]

        criterion = self.det_losses[sample_list.dataset_name]
        loss_dict = criterion(detr_outputs, sample_list.detr_target)
        weight_dict = criterion.weight_dict
        losses = {
            f"{sample_list.dataset_type}/{sample_list.dataset_name}/{k}": loss_dict[k]
            * weight_dict[k]
            * self.config.detection_loss_weight
            for k in loss_dict.keys()
            if k in weight_dict
        }
        detr_outputs["losses"] = losses
        # detr_outputs["targets"] = []  # dummy output for VQA accuracy gathering
        # detr_outputs["scores"] = []  # dummy output for VQA accuracy gathering
        return detr_outputs

    def classifier_loss_calculation(self, detr_outputs, sample_list):
        task_type = self.get_task_type(sample_list.dataset_name)
        hs = detr_outputs["hidden_states"]
        if not self.config.loss_on_all_hs[task_type][sample_list.dataset_name]:
            hs = detr_outputs["hidden_states"][-1:]
        num_queries = self.config.detr_args.num_queries[task_type][
            sample_list.dataset_name
        ]
        assert hs[0].size(1) == num_queries
        losses = {}
        scores = None
        detr_outputs = {}
        num_labels = self.config.datasets[task_type][sample_list.dataset_name][
            "num_labels"
        ]

        for idx, current_hs in enumerate(hs):
            pooled_output = hs[idx][:, -num_queries, :]
            pooled_output = self.dropout(pooled_output)
            logits = self.classifiers[task_type][sample_list.dataset_name](
                pooled_output
            )
            reshaped_logits = logits.contiguous().view(-1, num_labels)
            scores = reshaped_logits
            key = f"{sample_list.dataset_type}/{sample_list.dataset_name}/loss_{idx}"
            loss = self.losses_dict[task_type][sample_list.dataset_name](
                scores, sample_list.targets
            )
            if sample_list.dataset_name == "vqa2":
                loss *= sample_list.targets.size(1)
            losses[key] = loss

        detr_outputs["scores"] = scores
        detr_outputs["losses"] = losses
        return detr_outputs

    def get_optimizer_parameters(self, config):
        detr_params = [
            {
                "params": [
                    p
                    for n, p in self.detr_model.named_parameters()
                    if "backbone" not in n and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.detr_model.named_parameters()
                    if "backbone" in n and p.requires_grad
                ],
                "lr": self.config.detr_args.lr_backbone,
            },
        ]

        vqa_params = [
            {"params": self.bert_model.parameters()},
            {"params": self.bert_projection.parameters()},
            {"params": self.task_embeddings_lang.parameters()},
            {"params": self.bert_pos_projection.parameters()},
            {"params": self.classifiers.parameters()},
            {"params": self.class_embeds.parameters()},
            {"params": self.bbox_embeds.parameters()},
            {"params": self.det_losses.parameters()},
        ]
        return vqa_params + detr_params

    def get_task_idx(self, dataset_name):
        task_type = self.get_task_type(dataset_name)
        assert task_type in self.config.datasets
        return self.config.datasets[task_type][dataset_name]["task_idx"]

    def get_task_type(self, dataset_name):
        task_type = "detection"
        if dataset_name in self.config.datasets["vl"]:
            task_type = "vl"
        elif dataset_name in self.config.datasets["glue"]:
            task_type = "glue"
        return task_type
