# Copyright (c) Facebook, Inc. and its affiliates.
import csv
import json
import logging
import os

from mmf.common.batch_collator import BatchCollator
from mmf.common.registry import registry
from mmf.utils.build import build_dataloader_and_sampler
from mmf.utils.configuration import get_mmf_env
from mmf.utils.distributed import (
    gather_tensor,
    gather_tensor_along_batch,
    is_dist_initialized,
    is_master,
)
from mmf.utils.file_io import PathManager
from mmf.utils.general import (
    ckpt_name_from_core_args,
    foldername_from_config_override,
    get_batch_size,
)
from mmf.utils.timer import Timer
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


logger = logging.getLogger(__name__)


class TestReporter(Dataset):
    def __init__(self, multi_task_instance):
        self.test_task = multi_task_instance
        self.task_type = multi_task_instance.dataset_type
        self.config = registry.get("config")
        self.report = []
        self.timer = Timer()
        self.training_config = self.config.training
        self.num_workers = self.training_config.num_workers
        self.batch_size = self.training_config.batch_size
        self.report_folder_arg = get_mmf_env(key="report_dir")
        self.experiment_name = self.training_config.experiment_name

        self.datasets = []

        for dataset in self.test_task.get_datasets():
            self.datasets.append(dataset)

        self.current_dataset_idx = -1
        self.current_dataset = self.datasets[self.current_dataset_idx]

        self.save_dir = get_mmf_env(key="save_dir")
        self.report_folder = ckpt_name_from_core_args(self.config)
        self.report_folder += foldername_from_config_override(self.config)

        self.report_folder = os.path.join(self.save_dir, self.report_folder)
        self.report_folder = os.path.join(self.report_folder, "reports")

        if self.report_folder_arg:
            self.report_folder = self.report_folder_arg

        PathManager.mkdirs(self.report_folder)

    def next_dataset(self, flush_report=True):
        if self.current_dataset_idx >= 0:
            if flush_report:
                self.flush_report()
            else:
                self.report = []

        self.current_dataset_idx += 1

        if self.current_dataset_idx == len(self.datasets):
            return False
        else:
            self.current_dataset = self.datasets[self.current_dataset_idx]
            logger.info(f"Predicting for {self.current_dataset.dataset_name}")
            return True

    def flush_report(self):
        if not is_master():
            return

        name = self.current_dataset.dataset_name
        time_format = "%Y-%m-%dT%H:%M:%S"
        time = self.timer.get_time_hhmmss(None, format=time_format)

        filename = name + "_"

        if len(self.experiment_name) > 0:
            filename += self.experiment_name + "_"

        filename += self.task_type + "_"
        filename += time

        if self.config.evaluation.predict_file_format == "csv":
            filepath = os.path.join(self.report_folder, filename + ".csv")
            self.csv_dump(filepath)
        else:
            filepath = os.path.join(self.report_folder, filename + ".json")
            self.json_dump(filepath)

        logger.info(f"Wrote predictions for {name} to {os.path.abspath(filepath)}")
        self.report = []

    def postprocess_dataset_report(self):
        if hasattr(self.current_dataset, "on_prediction_end"):
            self.report = self.current_dataset.on_prediction_end(self.report)

    def csv_dump(self, filepath):
        with PathManager.open(filepath, "w") as f:
            title = self.report[0].keys()
            cw = csv.DictWriter(f, title, delimiter=",", quoting=csv.QUOTE_MINIMAL)
            cw.writeheader()
            cw.writerows(self.report)

    def json_dump(self, filepath):
        with PathManager.open(filepath, "w") as f:
            json.dump(self.report, f)

    def get_dataloader(self):
        dataloader, _ = build_dataloader_and_sampler(
            self.current_dataset, self.training_config
        )
        return dataloader

    def prepare_batch(self, batch):
        if hasattr(self.current_dataset, "prepare_batch"):
            batch = self.current_dataset.prepare_batch(batch)
        return batch

    def __len__(self):
        return len(self.current_dataset)

    def __getitem__(self, idx):
        return self.current_dataset[idx]

    def add_to_report(self, report, model, master_only=True):
        keys = ["id", "question_id", "image_id", "context_tokens", "captions", "scores"]
        for key in keys:
            report = self.reshape_and_gather(report, key)

        # gather detection output keys
        if hasattr(report, "pred_boxes"):
            report.pred_boxes = gather_tensor_along_batch(report.pred_boxes)
            report.pred_logits = gather_tensor_along_batch(report.pred_logits)
            report.orig_size = gather_tensor_along_batch(report.orig_size)

        if master_only and not is_master():
            return

        results = self.current_dataset.format_for_prediction(report)

        if hasattr(model, "format_for_prediction"):
            results = model.format_for_prediction(results, report)
        elif hasattr(model.module, "format_for_prediction"):
            results = model.module.format_for_prediction(results, report)

        self.report = self.report + results

    def reshape_and_gather(self, report, key):
        if key in report:
            num_dims = report[key].dim()
            if num_dims == 1:
                report[key] = gather_tensor(report[key]).view(-1)
            elif num_dims >= 2:
                # Collect dims other than batch
                other_dims = report[key].size()[1:]
                report[key] = gather_tensor(report[key]).view(-1, *other_dims)

        return report
