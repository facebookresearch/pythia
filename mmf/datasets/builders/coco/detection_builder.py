# Copyright (c) Facebook, Inc. and its affiliates.
from mmf.common.registry import registry
from mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder

from .detection_dataset import DetectionCOCODataset


@registry.register_builder("detection_coco")
class DetectionCOCOBuilder(MMFDatasetBuilder):
    def __init__(self):
        super().__init__(
            dataset_name="detection_coco", dataset_class=DetectionCOCODataset
        )

    @classmethod
    def config_path(cls):
        return "configs/datasets/coco/detection.yaml"

    def update_registry_for_model(self, config):
        pass
