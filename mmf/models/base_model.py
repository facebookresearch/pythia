# Copyright (c) Facebook, Inc. and its affiliates.

"""
Models built in MMF need to inherit ``BaseModel`` class and adhere to
a fixed format. To create a model for MMF, follow this quick cheatsheet.

1. Inherit ``BaseModel`` class, make sure to call ``super().__init__()`` in your
   class's ``__init__`` function.
2. Implement `build` function for your model. If you build everything in ``__init__``,
   you can just return in this function.
3. Write a `forward` function which takes in a ``SampleList`` as an argument and
   returns a dict.
4. Register using ``@registry.register_model("key")`` decorator on top of the
   class.

If you are doing logits based predictions, the dict you return from your model
should contain a `scores` field. Losses are automatically calculated by the
``BaseModel`` class and added to this dict if not present.

Example::

    import torch

    from mmf.common.registry import registry
    from mmf.models.base_model import BaseModel


    @registry.register("pythia")
    class Pythia(BaseModel):
        # config is model_config from global config
        def __init__(self, config):
            super().__init__(config)

        def build(self):
            ....

        def forward(self, sample_list):
            scores = torch.rand(sample_list.get_batch_size(), 3127)
            return {"scores": scores}
"""


import collections
import logging
import warnings
from copy import deepcopy
from dataclasses import dataclass
from typing import Union

import pytorch_lightning as pl
from mmf.common.registry import registry
from mmf.common.sample import SampleList, to_device
from mmf.modules.losses import Losses
from mmf.utils.checkpoint import load_pretrained_model
from mmf.utils.download import download_pretrained_model
from mmf.utils.file_io import PathManager
from omegaconf import MISSING, DictConfig, OmegaConf


try:
    import pytorch_lightning as pl
except ImportError:
    print(
        "BaseModel requires Pytorch Lightning. "
        + "Please follow the installation here: "
        + "https://pytorch-lightning.readthedocs.io/"
        + "en/latest/introduction_guide.html"
    )
    raise

logger = logging.getLogger(__name__)


class BaseModel(pl.LightningModule):
    """For integration with MMF's trainer, datasets and other features,
    models needs to inherit this class, call `super`, write a build function,
    write a forward function taking a ``SampleList`` as input and returning a
    dict as output and finally, register it using ``@registry.register_model``

    Args:
        config (DictConfig): ``model_config`` configuration from global config.

    """

    @dataclass
    class Config:
        # Name of the model that is used in registry
        model: str = MISSING

    def __init__(self, config: Union[DictConfig, Config]):
        super().__init__()
        if not isinstance(config, DictConfig) and isinstance(config, self.Config):
            config = OmegaConf.structured(config)

        self.config = config

        self._logged_warning = {"losses_present": False}
        self._is_pretrained = False
        self._is_pl_enabled = False

    @classmethod
    def from_params(cls, **kwargs):
        return cls(OmegaConf.structured(cls.Config(**kwargs)))

    @property
    def is_pretrained(self):
        return self._is_pretrained

    @property
    def is_pl_enabled(self):
        return self._is_pl_enabled

    @is_pretrained.setter
    def is_pretrained(self, x: bool):
        self._is_pretrained = x

    @is_pl_enabled.setter
    def is_pl_enabled(self, x: bool):
        self._is_pl_enabled = x

    def build(self):
        """Function to be implemented by the child class, in case they need to
        build their model separately than ``__init__``. All model related
        downloads should also happen here.
        """
        raise NotImplementedError(
            "Build method not implemented in the child model class."
        )

    def init_losses(self):
        """Initializes loss for the model based ``losses`` key. Automatically called by
        MMF internally after building the model.
        """
        losses = self.config.get("losses", [])
        if len(losses) == 0 and not self.is_pretrained:
            warnings.warn(
                "No losses are defined in model configuration. You are expected "
                "to return loss in your return dict from forward."
            )

        self.losses = Losses(losses)

    @classmethod
    def config_path(cls):
        return None

    @classmethod
    def format_state_key(cls, key):
        """Can be implemented if something special needs to be done to the
        key when pretrained model is being loaded. This will adapt and return
        keys according to that. Useful for backwards compatibility. See
        updated load_state_dict below. For an example, see VisualBERT model's
        code.

        Args:
            key (string): key to be formatted

        Returns:
            string: formatted key
        """
        return key

    def load_state_dict(self, state_dict, *args, **kwargs):
        copied_state_dict = deepcopy(state_dict)
        for key in list(copied_state_dict.keys()):
            formatted_key = self.format_state_key(key)
            copied_state_dict[formatted_key] = copied_state_dict.pop(key)

        return super().load_state_dict(copied_state_dict, *args, **kwargs)

    def forward(self, sample_list, *args, **kwargs):
        """To be implemented by child class. Takes in a ``SampleList`` and
        returns back a dict.

        Args:
            sample_list (SampleList): SampleList returned by the DataLoader for
            current iteration

        Returns:
            Dict: Dict containing scores object.

        """
        raise NotImplementedError(
            "Forward of the child model class needs to be implemented."
        )

    def training_step(self, batch, batch_idx, *args, **kwargs):
        """Member function of PL modules. Used only when PL enabled.
        To be implemented by child class. Takes in a ``SampleList``,
        batch_idx and returns back a dict.

        Args:
            sample_list (SampleList): SampleList returned by the DataLoader for
            current iteration

        Returns:
            Dict: Dict containing loss.
        """
        return self._forward_lightning_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        """Member function of PL modules. Used only when PL enabled.
        To be implemented by child class. Takes in a ``SampleList``,
        batch_idx and returns back a dict.

        Args:
            sample_list (SampleList): SampleList returned by the DataLoader for
            current iteration

        Returns:
            Dict
        """
        return self._forward_lightning_step(batch, batch_idx)

    def _forward_lightning_step(self, batch, batch_idx):
        batch = self._ensure_sample_list(batch)
        output = self(batch)
        loss_dict = output["losses"]
        output["loss"] = sum(loss.mean() for loss in loss_dict.values())
        output["input_batch"] = batch
        return output

    def configure_optimizers(self):
        """ Member function of PL modules. Used only when PL enabled."""
        assert self._is_pl_enabled, (
            "configure_optimizers should be only used as a member "
            "function of LightningModule when pytorch lightning is enabled."
        )

        from mmf.utils.build import build_lightning_optimizers

        config = registry.get("config")
        return build_lightning_optimizers(self, config)

    def _ensure_sample_list(self, batch):
        if not isinstance(batch, SampleList):
            # Try converting to SampleList
            batch = SampleList(batch)
        return batch

    def __call__(self, sample_list, *args, **kwargs):
        if not self._is_pl_enabled:
            # Move to proper device i.e. same as the model before passing
            model_device = next(self.parameters()).device
            sample_list = to_device(sample_list, model_device)

        model_output = super().__call__(sample_list, *args, **kwargs)

        # Don't do anything fancy to output if it is pretrained
        if self.is_pretrained:
            return model_output

        # Make sure that the output from the model is a Mapping
        assert isinstance(
            model_output, collections.abc.Mapping
        ), "A dict must be returned from the forward of the model."

        if "losses" in model_output:
            if not self._logged_warning["losses_present"]:
                warnings.warn(
                    "'losses' already present in model output. "
                    "No calculation will be done in base model."
                )
                self._logged_warning["losses_present"] = True

            assert isinstance(
                model_output["losses"], collections.abc.Mapping
            ), "'losses' must be a dict."
        elif hasattr(self, "losses"):
            model_output["losses"] = self.losses(sample_list, model_output)
        else:
            model_output["losses"] = {}

        return model_output

    def load_requirements(self, *args, **kwargs):
        requirements = self.config.get("zoo_requirements", [])
        if isinstance(requirements, str):
            requirements = [requirements]
        for item in requirements:
            download_pretrained_model(item, *args, **kwargs)

    def format_for_prediction(self, results, report):
        """Implement this method in models if it requires to modify prediction
        results using report fields. Note that the required fields in report
        should already be gathered in report.
        """
        return results

    @classmethod
    def from_pretrained(cls, model_name_or_path, *args, **kwargs):
        # Check if the path exists, if not it is pretrained, otherwise,
        # we will try to load the checkpoint from the path
        if not PathManager.exists(model_name_or_path):
            model_key = model_name_or_path.split(".")[0]
            model_cls = registry.get_model_class(model_key)
            assert (
                model_cls == cls
            ), f"Incorrect pretrained model key {model_name_or_path} "
            "for class {cls.__name__}"
        output = load_pretrained_model(model_name_or_path, *args, **kwargs)
        config, checkpoint, full_config = (
            output["config"],
            output["checkpoint"],
            output["full_config"],
        )

        # Save original config for state reset later
        config_temp_holder = registry.get("config")
        # Register full config from checkpoint when loading the model
        registry.register("config", full_config)

        # Some models need registry updates to be load pretrained model
        # If they have this method, call it so they can update accordingly
        if hasattr(cls, "update_registry_for_pretrained"):
            cls.update_registry_for_pretrained(config, checkpoint, output)

        instance = cls(config)
        instance.is_pretrained = True
        instance.build()
        incompatible_keys = instance.load_state_dict(checkpoint, strict=False)

        # The model has loaded, reset the state
        registry.register("config", config_temp_holder)

        if len(incompatible_keys.missing_keys) != 0:
            logger.warning(
                f"Missing keys {incompatible_keys.missing_keys} in the"
                + " checkpoint.\n"
                + "If this is not your checkpoint, please open up an "
                + "issue on MMF GitHub. \n"
                + f"Unexpected keys if any: {incompatible_keys.unexpected_keys}"
            )

        if len(incompatible_keys.unexpected_keys) != 0:
            logger.warning(
                "Unexpected keys in state dict: "
                + f"{incompatible_keys.unexpected_keys} \n"
                + "This is usually not a problem with pretrained models, but "
                + "if this is your own model, please double check. \n"
                + "If you think this is an issue, please open up a "
                + "bug at MMF GitHub."
            )

        instance.eval()

        return instance
