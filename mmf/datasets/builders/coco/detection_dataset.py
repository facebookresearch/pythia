# Copyright (c) Facebook, Inc. and its affiliates.
import os

import torch
import torch.nn.functional as F
import torchvision
from mmf.common.sample import Sample
from mmf.datasets.base_dataset import BaseDataset
from mmf.modules.detr.datasets.coco import ConvertCocoPolysToMask, make_coco_transforms
from torch import nn


class DetectionCOCODataset(BaseDataset):
    def __init__(self, config, dataset_type, imdb_file_index, *args, **kwargs):
        name = "detection_coco"
        super().__init__(name, config, dataset_type, *args, **kwargs)
        self.dataset_name = name

        self.image_dir = self.config.images[self._dataset_type][imdb_file_index]
        coco_json = self.config.annotations[self._dataset_type][imdb_file_index]
        self.coco_json = os.path.join(self.config.data_dir, coco_json)

        self.coco_dataset = torchvision.datasets.CocoDetection(
            self.image_dir, self.coco_json
        )

        self.prepare = ConvertCocoPolysToMask()
        # for the test set, use the same transform as val
        self.transform = make_coco_transforms(
            "train" if self._dataset_type == "train" else "val"
        )
        self.load_attributes = self.config.load_attributes
        self.postprocessors = {"bbox": PostProcess()}

    def __getitem__(self, idx):
        img, target = self.coco_dataset[idx]
        image_id = self.coco_dataset.ids[idx]
        target = {"image_id": image_id, "annotations": target}
        img, target = self.prepare(img, target, load_attributes=self.load_attributes)
        img, target = self.transform(img, target)

        current_sample = Sample()
        current_sample.image_id = torch.tensor(image_id, dtype=torch.long)
        current_sample.detr_img = img
        current_sample.detr_target = target
        current_sample.orig_size = target["orig_size"].clone().detach()

        return current_sample

    def __len__(self):
        return len(self.coco_dataset)

    def prepare_batch(self, batch):
        batch = super().prepare_batch(batch)
        # batch.detr_target is a list and won't be sent to GPU automatically
        # so we manually send it to GPU here
        batch.detr_target = [
            {k: v.to(self._device) for k, v in t.items()} for t in batch.detr_target
        ]
        return batch

    def format_for_prediction(self, report):
        outputs = {"pred_logits": report.pred_logits, "pred_boxes": report.pred_boxes}

        image_ids = report.image_id.tolist()
        results = self.postprocessors["bbox"](outputs, report.orig_size)

        predictions = []
        for image_id, r in zip(image_ids, results):
            scores = r["scores"].tolist()
            labels = r["labels"].tolist()
            boxes_xywh = convert_to_xywh(r["boxes"]).tolist()

            # group the boxes by image_id for image-level de-duplication
            # (duplication is introduced by DistributedSampler)
            predictions.append(
                (
                    image_id,
                    [
                        {
                            "image_id": image_id,
                            "category_id": labels[k],
                            "bbox": box_xywh,
                            "score": scores[k],
                        }
                        for k, box_xywh in enumerate(boxes_xywh)
                    ],
                )
            )

        return predictions

    def on_prediction_end(self, predictions):
        # de-duplicate the predictions (duplication is introduced by DistributedSampler)
        prediction_dict = {image_id: entries for image_id, entries in predictions}

        unique_entries = []
        for image_id in sorted(prediction_dict):
            unique_entries.extend(prediction_dict[image_id])

        return unique_entries


class PostProcess(nn.Module):
    """This module converts the model's output into the format expected by the
    coco api
    """

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of
                          each images of the batch For evaluation, this must be the
                          original image size (before any data augmentation). For
                          visualization, this should be the image size after data
                          augment, but before padding.
        """
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        from mmf.modules.detr.util.box_ops import box_cxcywh_to_xyxy

        boxes = box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [
            {"scores": s, "labels": l, "boxes": b}
            for s, l, b in zip(scores, labels, boxes)
        ]

        return results


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)
