# Copyright (c) Facebook, Inc. and its affiliates.

import math
import warnings
from typing import Any, Dict

import torch


def get_max_updates(config_max_updates, config_max_epochs, train_loader, update_freq):
    if config_max_updates is None and config_max_epochs is None:
        raise ValueError("Neither max_updates nor max_epochs is specified.")

    if isinstance(train_loader.current_dataset, torch.utils.data.IterableDataset):
        warnings.warn(
            "max_epochs not supported for Iterable datasets. Falling back "
            + "to max_updates."
        )
        return config_max_updates, config_max_epochs

    if config_max_updates is not None and config_max_epochs is not None:
        warnings.warn(
            "Both max_updates and max_epochs are specified. "
            + f"Favoring max_epochs: {config_max_epochs}"
        )

    if config_max_epochs is not None:
        max_updates = math.ceil(len(train_loader) / update_freq) * config_max_epochs
        max_epochs = config_max_epochs
    else:
        max_updates = config_max_updates
        max_epochs = max_updates / len(train_loader)

    return max_updates, max_epochs


def extract_loss(report: Dict[str, Any], loss_divisor: int) -> torch.Tensor:
    loss_dict = report.losses
    assert len(loss_dict) != 0, (
        "Model returned an empty loss dict. "
        "Did you forget to (i) define losses in your model configuration or"
        "(ii) return losses dict from your model?"
    )

    # Since losses are batch averaged in MMF, this makes sure the
    # scaling is right.
    for key, value in loss_dict.items():
        value = value.mean() / loss_divisor
        report.losses[key] = value

    loss = sum(loss.mean() for loss in loss_dict.values())
    return loss
