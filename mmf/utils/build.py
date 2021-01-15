# Copyright (c) Facebook, Inc. and its affiliates.

import os
import warnings
from typing import Any, Dict, Type, Union

import mmf
import torch
from mmf.common import typings as mmf_typings
from mmf.common.registry import registry
from mmf.datasets.processors.processors import Processor
from mmf.utils.configuration import Configuration
from mmf.utils.distributed import is_dist_initialized
from mmf.utils.general import get_optimizer_parameters
from omegaconf import DictConfig, OmegaConf


ProcessorType = Type[Processor]
ProcessorDict = Dict[str, ProcessorType]


def build_config(
    configuration: Type[Configuration], *args, **kwargs
) -> mmf_typings.DictConfig:
    """Builder function for config. Freezes the configuration and registers
    configuration object and config DictConfig object to registry.

    Args:
        configuration (Configuration): Configuration object that will be
            used to create the config.

    Returns:
        (DictConfig): A config which is of type Omegaconf.DictConfig
    """
    configuration.freeze()
    config = configuration.get_config()
    registry.register("config", config)
    registry.register("configuration", configuration)

    return config


def build_trainer(config: mmf_typings.DictConfig) -> Any:
    """Builder function for creating a trainer class. Trainer class name
    is picked from the config.

    Args:
        config (DictConfig): Configuration that will be used to create
            the trainer.

    Returns:
        (BaseTrainer): A trainer instance
    """
    trainer_type = config.training.trainer
    trainer_cls = registry.get_trainer_class(trainer_type)
    trainer_obj = trainer_cls(config)

    return trainer_obj


def build_model(
    config: Union[DictConfig, "mmf.models.base_model.BaseModel.Config"]
) -> "mmf.models.base_model.BaseModel":
    from mmf.models.base_model import BaseModel

    # If it is not an OmegaConf object, create the object
    if not isinstance(config, DictConfig) and isinstance(config, BaseModel.Config):
        config = OmegaConf.structured(config)

    model_name = config.model
    model_class = registry.get_model_class(model_name)

    if model_class is None:
        raise RuntimeError(f"No model registered for name: {model_name}")
    model = model_class(config)

    if hasattr(model, "build"):
        model.load_requirements()
        model.build()
        model.init_losses()

    return model


def build_dataset(
    dataset_key: str, config=None, dataset_type="train"
) -> mmf_typings.DatasetType:
    """Builder function for creating a dataset. If dataset_key is passed
    the dataset is created from default config of the dataset and thus is
    disable config even if it is passed. Otherwise, we use MultiDatasetLoader to
    build and return an instance of dataset based on the config

    Args:
        dataset_key (str): Key of dataset to build.
        config (DictConfig, optional): Configuration that will be used to create
            the dataset. If not passed, dataset's default config will be used.
            Defaults to {}.
        dataset_type (str, optional): Type of the dataset to build, train|val|test.
            Defaults to "train".

    Returns:
        (DatasetType): A dataset instance of type BaseDataset
    """
    from mmf.utils.configuration import load_yaml_with_defaults

    dataset_builder = registry.get_builder_class(dataset_key)
    assert dataset_builder, (
        f"Key {dataset_key} doesn't have a registered " + "dataset builder"
    )

    # If config is not provided, we take it from default one
    if not config:
        config = load_yaml_with_defaults(dataset_builder.config_path())
        config = OmegaConf.select(config, f"dataset_config.{dataset_key}")
        OmegaConf.set_struct(config, True)

    builder_instance: mmf_typings.DatasetBuilderType = dataset_builder()
    builder_instance.build_dataset(config, dataset_type)
    dataset = builder_instance.load_dataset(config, dataset_type)
    if hasattr(builder_instance, "update_registry_for_model"):
        builder_instance.update_registry_for_model(config)

    return dataset


def build_dataloader_and_sampler(
    dataset_instance: mmf_typings.DatasetType, training_config: mmf_typings.DictConfig
) -> mmf_typings.DataLoaderAndSampler:
    """Builds and returns a dataloader along with its sample

    Args:
        dataset_instance (mmf_typings.DatasetType): Instance of dataset for which
            dataloader has to be created
        training_config (mmf_typings.DictConfig): Training configuration; required
            for infering params for dataloader

    Returns:
        mmf_typings.DataLoaderAndSampler: Tuple of Dataloader and Sampler instance
    """
    from mmf.common.batch_collator import BatchCollator

    num_workers = training_config.num_workers
    pin_memory = training_config.pin_memory

    other_args = {}

    # IterableDataset returns batches directly, so no need to add Sampler
    # or batch size as user is expected to control those. This is a fine
    # assumption for now to not support single item based IterableDataset
    # as it will add unnecessary complexity and config parameters
    # to the codebase
    if not isinstance(dataset_instance, torch.utils.data.IterableDataset):
        other_args = _add_extra_args_for_dataloader(dataset_instance, other_args)

    loader = torch.utils.data.DataLoader(
        dataset=dataset_instance,
        pin_memory=pin_memory,
        collate_fn=BatchCollator(
            dataset_instance.dataset_name, dataset_instance.dataset_type
        ),
        num_workers=num_workers,
        drop_last=False,  # see also MultiDatasetLoader.__len__
        **other_args,
    )

    if num_workers >= 0:
        # Suppress leaking semaphore warning
        os.environ["PYTHONWARNINGS"] = "ignore:semaphore_tracker:UserWarning"

    loader.dataset_type = dataset_instance.dataset_type

    return loader, other_args.get("sampler", None)


def _add_extra_args_for_dataloader(
    dataset_instance: mmf_typings.DatasetType,
    other_args: mmf_typings.DataLoaderArgsType = None,
) -> mmf_typings.DataLoaderArgsType:
    from mmf.utils.general import get_batch_size

    if other_args is None:
        other_args = {}
    dataset_type = dataset_instance.dataset_type

    other_args["shuffle"] = False
    if dataset_type != "test":
        other_args["shuffle"] = True

    # In distributed mode, we use DistributedSampler from PyTorch
    if is_dist_initialized():
        other_args["sampler"] = torch.utils.data.DistributedSampler(
            dataset_instance, shuffle=other_args["shuffle"]
        )
        # Shuffle is mutually exclusive with sampler, let DistributedSampler
        # take care of shuffle and pop from main args
        other_args.pop("shuffle")

    other_args["batch_size"] = get_batch_size()

    return other_args


def build_optimizer(model, config):
    optimizer_config = config.optimizer
    if not hasattr(optimizer_config, "type"):
        raise ValueError(
            "Optimizer attributes must have a 'type' key "
            "specifying the type of optimizer. "
            "(Custom or PyTorch)"
        )
    optimizer_type = optimizer_config.type

    if not hasattr(optimizer_config, "params"):
        warnings.warn("optimizer attributes has no params defined, defaulting to {}.")

    params = getattr(optimizer_config, "params", {})

    if hasattr(torch.optim, optimizer_type):
        optimizer_class = getattr(torch.optim, optimizer_type)
    else:
        optimizer_class = registry.get_optimizer_class(optimizer_type)
        if optimizer_class is None:
            raise ValueError(
                "No optimizer class of type {} present in "
                "either torch or registered to registry"
            )

    parameters = get_optimizer_parameters(model, config)

    if optimizer_config.get("enable_state_sharding", False):
        # TODO(vedanuj): Remove once OSS is moved to PT upstream
        try:
            from fairscale.optim.oss import OSS
        except ImportError:
            print(
                "Optimizer state sharding requires fairscale. "
                + "Install using pip install fairscale."
            )
            raise

        assert (
            is_dist_initialized()
        ), "Optimizer state sharding can only be used in distributed mode."
        optimizer = OSS(params=parameters, optim=optimizer_class, **params)
    else:
        optimizer = optimizer_class(parameters, **params)
    return optimizer

def build_lightning_optimizers(model, config):
    optimizer = build_optimizer(model, config)

    if config.training.lr_scheduler:
        lr_scheduler = build_scheduler(optimizer, config)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step"},
        }
    else:
        return optimizer


def build_lightning_optimizers(model, config):
    optimizer = build_optimizer(model, config)

    if config.training.lr_scheduler:
        lr_scheduler = build_scheduler(optimizer, config)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step"},
        }
    else:
        return optimizer


def build_scheduler(optimizer, config):
    scheduler_config = config.get("scheduler", {})

    if not hasattr(scheduler_config, "type"):
        warnings.warn(
            "No type for scheduler specified even though lr_scheduler is True, "
            "setting default to 'Pythia'"
        )
    scheduler_type = getattr(scheduler_config, "type", "pythia")

    if not hasattr(scheduler_config, "params"):
        warnings.warn("scheduler attributes has no params defined, defaulting to {}.")
    params = getattr(scheduler_config, "params", {})
    scheduler_class = registry.get_scheduler_class(scheduler_type)
    scheduler = scheduler_class(optimizer, **params)

    return scheduler


def build_classifier_layer(config, *args, **kwargs):
    from mmf.modules.layers import ClassifierLayer

    classifier = ClassifierLayer(config.type, *args, **config.params, **kwargs)
    return classifier.module


def build_text_encoder(config, *args, **kwargs):
    try:
        from mmf.modules.fb.encoders import TextEncoderFactory
    except ImportError:
        from mmf.modules.encoders import TextEncoderFactory

    text_encoder = TextEncoderFactory(config, *args, **kwargs)
    return text_encoder.module


def build_image_encoder(config, direct_features=False, **kwargs):
    from mmf.modules.encoders import ImageEncoderFactory, ImageFeatureEncoderFactory

    if direct_features:
        module = ImageFeatureEncoderFactory(config)
    else:
        module = ImageEncoderFactory(config)
    return module.module


def build_encoder(config: Union[DictConfig, "mmf.modules.encoders.Encoder.Config"]):
    from mmf.modules.encoders import Encoder

    # If it is not an OmegaConf object, create the object
    if not isinstance(config, DictConfig) and isinstance(config, Encoder.Config):
        config = OmegaConf.structured(config)

    if "type" in config:
        # Support config initialization in form of
        # encoder:
        #   type: identity # noqa
        #   params:
        #       in_dim: 256
        name = config.type
        params = config.params
    else:
        # Structured Config support
        name = config.name
        params = config

    encoder_cls = registry.get_encoder_class(name)
    return encoder_cls(params)


def build_processors(
    processors_config: mmf_typings.DictConfig, registry_key: str = None, *args, **kwargs
) -> ProcessorDict:
    """Given a processor config, builds the processors present and returns back
    a dict containing processors mapped to keys as per the config

    Args:
        processors_config (mmf_typings.DictConfig): OmegaConf DictConfig describing
            the parameters and type of each processor passed here

        registry_key (str, optional): If passed, function would look into registry for
            this particular key and return it back. .format with processor_key will
            be called on this string. Defaults to None.

    Returns:
        ProcessorDict: Dictionary containing key to
            processor mapping
    """
    from mmf.datasets.processors.processors import Processor

    processor_dict = {}

    for processor_key, processor_params in processors_config.items():
        if not processor_params:
            continue

        processor_instance = None
        if registry_key is not None:
            full_key = registry_key.format(processor_key)
            processor_instance = registry.get(full_key, no_warning=True)

        if processor_instance is None:
            processor_instance = Processor(processor_params, *args, **kwargs)
            # We don't register back here as in case of hub interface, we
            # want the processors to be instantiate every time. BaseDataset
            # can register at its own end
        processor_dict[processor_key] = processor_instance

    return processor_dict
