import copy
import json
import os

import torch
from mmf.common.sample import Sample
from mmf.datasets.builders.vqa2 import VQA2Dataset


LABEL_TO_INT_MAPPING = {"entailment": 0, "neutral": 1, "contradiction": 2}


class VisualEntailmentDataset(VQA2Dataset):
    def __init__(self, config, dataset_type, imdb_file_index, *args, **kwargs):
        super().__init__(
            config,
            dataset_type,
            imdb_file_index,
            dataset_name="visual_entailment",
            *args,
            **kwargs
        )

    def load_item(self, idx):
        sample_info = self.annotation_db[idx]
        current_sample = Sample()

        processed_sentence = self.text_processor({"text": sample_info["sentence2"]})

        current_sample.text = processed_sentence["text"]
        if "input_ids" in processed_sentence:
            current_sample.update(processed_sentence)

        if self._use_features is True:
            # Remove sentence id from end
            identifier = sample_info["Flikr30kID"].split(".")[0]
            # Load img0 and img1 features
            sample_info["feature_path"] = "{}.npy".format(identifier)
            features = self.features_db[idx]
            if hasattr(self, "transformer_bbox_processor"):
                features["image_info_0"] = self.transformer_bbox_processor(
                    features["image_info_0"]
                )
            current_sample.update(features)

        label = LABEL_TO_INT_MAPPING[sample_info["gold_label"]]
        current_sample.targets = torch.tensor(label, dtype=torch.long)

        if self.load_detr_img:
            image_path = os.path.join(self.image_dir, sample_info["Flikr30kID"])
            current_sample.detr_img = self.detr_image_processor(
                {"image_path": image_path}
            )["detr_img"]

        return current_sample

    def format_for_prediction(self, report):
        return []
