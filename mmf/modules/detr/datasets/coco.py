# Copyright (c) Facebook, Inc. and its affiliates.

# Mostly copy-pasted from
# https://github.com/facebookresearch/detr/blob/master/datasets/coco.py
from pathlib import Path

import torch
import torch.utils.data
import torchvision

from pycocotools import mask as coco_mask

from . import transforms as T


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {"image_id": image_id, "annotations": target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target, load_attributes=False):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        attributes = None
        if load_attributes:
            MAX_ATTR_NUM = 16
            # -1 will be used as ignore label
            attributes = -torch.ones(len(classes), MAX_ATTR_NUM, dtype=torch.int64)
            for n_obj, obj in enumerate(anno):
                if "attribute_ids_max16" in obj:
                    attributes[n_obj] = torch.as_tensor(
                        obj["attribute_ids_max16"], dtype=torch.int64
                    )
                elif "attributes" in obj:
                    attr_ids = obj["attributes"]
                    if "attribute_names" not in obj:
                        # merged attributes, flatten the list
                        attr_ids = [a for attr_list in attr_ids for a in attr_list]
                    attr_ids = attr_ids[:MAX_ATTR_NUM]
                    attributes[n_obj, : len(attr_ids)] = torch.as_tensor(
                        attr_ids, dtype=torch.int64
                    )

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if attributes is not None:
            attributes = attributes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["orig_boxes"] = boxes
        target["labels"] = classes
        if attributes is not None:
            target["attributes"] = attributes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor(
            [obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno]
        )
        target["area"] = area[keep]
        target["orig_area"] = target["area"]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set):

    normalize = T.Compose(
        [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == "train":
        return T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=1333),
                    T.Compose(
                        [
                            T.RandomResize([400, 500, 600]),
                            T.RandomSizeCrop(384, 600),
                            T.RandomResize(scales, max_size=1333),
                        ]
                    ),
                ),
                normalize,
            ]
        )

    if image_set == "val":
        return T.Compose([T.RandomResize([800], max_size=1333), normalize])

    raise ValueError(f"unknown {image_set}")


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f"provided COCO path {root} does not exist"
    mode = "instances"
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f"{mode}_train2017.json"),
        "val": (root / "val2017", root / "annotations" / f"{mode}_val2017.json"),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(
        img_folder,
        ann_file,
        transforms=make_coco_transforms(image_set),
        return_masks=args.masks,
    )
    return dataset
