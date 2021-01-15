# Copyright (c) Facebook, Inc. and its affiliates.

import collections
import functools
import json
import logging
import os
import sys
import time
from typing import Any, Dict, Union

from mmf.common.registry import registry
from mmf.utils.configuration import get_mmf_env
from mmf.utils.distributed import get_rank, is_master
from mmf.utils.file_io import PathManager
from mmf.utils.timer import Timer
from termcolor import colored


@functools.lru_cache()
def setup_output_folder(folder_only: bool = False):
    """Sets up and returns the output file where the logs will be placed
    based on the configuration passed. Usually "save_dir/logs/log_<timestamp>.txt".
    If env.log_dir is passed, logs will be directly saved in this folder.

    Args:
        folder_only (bool, optional): If folder should be returned and not the file.
            Defaults to False.

    Returns:
        str: folder or file path depending on folder_only flag
    """
    save_dir = get_mmf_env(key="save_dir")
    time_format = "%Y_%m_%dT%H_%M_%S"
    log_filename = "train_"
    log_filename += Timer().get_time_hhmmss(None, format=time_format)
    log_filename += ".log"

    log_folder = os.path.join(save_dir, "logs")

    env_log_dir = get_mmf_env(key="log_dir")
    if env_log_dir:
        log_folder = env_log_dir

    if not PathManager.exists(log_folder):
        PathManager.mkdirs(log_folder)

    if folder_only:
        return log_folder

    log_filename = os.path.join(log_folder, log_filename)

    return log_filename


# so that calling setup_logger multiple times won't add many handlers
@functools.lru_cache()
def setup_logger(
    output: str = None,
    color: bool = True,
    name: str = "mmf",
    disable: bool = False,
    clear_handlers=True,
    *args,
    **kwargs,
):
    """
    Initialize the MMF logger and set its verbosity level to "INFO".
    Outside libraries shouldn't call this in case they have set there
    own logging handlers and setup. If they do, and don't want to
    clear handlers, pass clear_handlers options.

    The initial version of this function was taken from D2 and adapted
    for MMF.

    Args:
        output (str): a file name or a directory to save log.
            If ends with ".txt" or ".log", assumed to be a file name.
            Default: Saved to file <save_dir/logs/log_[timestamp].txt>
        color (bool): If false, won't log colored logs. Default: true
        name (str): the root module name of this logger. Defaults to "mmf".
        clear_handlers (bool): If false, won't clear existing handlers.

    Returns:
        logging.Logger: a logger
    """
    if disable:
        return None
    logger = logging.getLogger(name)
    logger.propagate = False

    logging.captureWarnings(True)
    warnings_logger = logging.getLogger("py.warnings")

    plain_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s : %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    distributed_rank = get_rank()
    handlers = []

    if distributed_rank == 0:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.INFO)
        if color:
            formatter = ColorfulFormatter(
                colored("%(asctime)s | %(name)s: ", "green") + "%(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        else:
            formatter = plain_formatter
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        warnings_logger.addHandler(ch)
        handlers.append(ch)

    # file logging: all workers
    if output is None:
        output = setup_output_folder()

    if output is not None:
        if output.endswith(".txt") or output.endswith(".log"):
            filename = output
        else:
            filename = os.path.join(output, "train.log")
        if distributed_rank > 0:
            filename = filename + f".rank{distributed_rank}"
        PathManager.mkdirs(os.path.dirname(filename))

        fh = logging.StreamHandler(_cached_log_stream(filename))
        fh.setLevel(logging.INFO)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)
        warnings_logger.addHandler(fh)
        handlers.append(fh)

        # Slurm/FB output, only log the main process
        if "train.log" not in filename and distributed_rank == 0:
            save_dir = get_mmf_env(key="save_dir")
            filename = os.path.join(save_dir, "train.log")
            sh = logging.StreamHandler(_cached_log_stream(filename))
            sh.setLevel(logging.INFO)
            sh.setFormatter(plain_formatter)
            logger.addHandler(sh)
            warnings_logger.addHandler(sh)
            handlers.append(sh)

        logger.info(f"Logging to: {filename}")

    # Remove existing handlers to add MMF specific handlers
    if clear_handlers:
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
    # Now, add our handlers.
    logging.basicConfig(level=logging.INFO, handlers=handlers)

    registry.register("writer", logger)

    return logger


def setup_very_basic_config(color=True):
    plain_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s : %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    if color:
        formatter = ColorfulFormatter(
            colored("%(asctime)s | %(name)s: ", "green") + "%(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    else:
        formatter = plain_formatter
    ch.setFormatter(formatter)
    # Setup a minimal configuration for logging in case something tries to
    # log a message even before logging is setup by MMF.
    logging.basicConfig(level=logging.INFO, handlers=[ch])


# cache the opened file object, so that different calls to `setup_logger`
# with the same file name can safely write to the same file.
@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    return PathManager.open(filename, "a")


def _find_caller():
    """
    Returns:
        str: module name of the caller
        tuple: a hashable key to be used to identify different callers
    """
    frame = sys._getframe(2)
    while frame:
        code = frame.f_code
        if os.path.join("utils", "logger.") not in code.co_filename:
            mod_name = frame.f_globals["__name__"]
            if mod_name == "__main__":
                mod_name = "mmf"
            return mod_name, (code.co_filename, frame.f_lineno, code.co_name)
        frame = frame.f_back


def summarize_report(
    current_iteration,
    num_updates,
    max_updates,
    meter,
    should_print=True,
    extra=None,
    tb_writer=None,
):
    if extra is None:
        extra = {}
    if not is_master():
        return

    if tb_writer:
        scalar_dict = meter.get_scalar_dict()
        tb_writer.add_scalars(scalar_dict, current_iteration)

    if not should_print:
        return
    log_dict = {}
    if num_updates and max_updates:
        log_dict.update({"progress": f"{num_updates}/{max_updates}"})
    log_dict.update(meter.get_log_dict())
    log_dict.update(extra)

    log_progress(log_dict)


def calculate_time_left(
    max_updates,
    num_updates,
    timer,
    num_snapshot_iterations,
    log_interval,
    eval_interval,
):
    time_taken_for_log = time.time() * 1000 - timer.start
    iterations_left = max_updates - num_updates
    num_logs_left = iterations_left / log_interval
    time_left = num_logs_left * time_taken_for_log

    snapshot_iteration = num_snapshot_iterations / log_interval
    snapshot_iteration *= iterations_left / eval_interval
    time_left += snapshot_iteration * time_taken_for_log

    return timer.get_time_hhmmss(gap=time_left)


def log_progress(info: Union[Dict, Any], log_format="simple"):
    """Useful for logging progress dict.

    Args:
        info (dict|any): If dict, will be logged as key value pair. Otherwise,
            it will be logged directly.

        log_format (str, optional): json|simple. Defaults to "simple".
            Will use simple mode.
    """
    caller, key = _find_caller()
    logger = logging.getLogger(caller)

    if not isinstance(info, collections.Mapping):
        logger.info(info)

    if log_format == "simple":
        config = registry.get("config")
        if config:
            log_format = config.training.log_format

    if log_format == "simple":
        output = ", ".join([f"{key}: {value}" for key, value in info.items()])
    elif log_format == "json":
        output = json.dumps(info)
    else:
        output = str(info)

    logger.info(output)


# ColorfulFormatter is adopted from Detectron2 and adapted for MMF
class ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def formatMessage(self, record):
        log = super().formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log


class TensorboardLogger:
    def __init__(self, log_folder="./logs", iteration=0):
        # This would handle warning of missing tensorboard
        from torch.utils.tensorboard import SummaryWriter

        self.summary_writer = None
        self._is_master = is_master()
        self.timer = Timer()
        self.log_folder = log_folder
        self.time_format = "%Y-%m-%dT%H:%M:%S"

        if self._is_master:
            current_time = self.timer.get_time_hhmmss(None, format=self.time_format)
            tensorboard_folder = os.path.join(
                self.log_folder, f"tensorboard_{current_time}"
            )
            self.summary_writer = SummaryWriter(tensorboard_folder)

    def __del__(self):
        if getattr(self, "summary_writer", None) is not None:
            self.summary_writer.close()

    def _should_log_tensorboard(self):
        if self.summary_writer is None or not self._is_master:
            return False
        else:
            return True

    def add_scalar(self, key, value, iteration):
        if not self._should_log_tensorboard():
            return

        self.summary_writer.add_scalar(key, value, iteration)

    def add_scalars(self, scalar_dict, iteration):
        if not self._should_log_tensorboard():
            return

        for key, val in scalar_dict.items():
            self.summary_writer.add_scalar(key, val, iteration)

    def add_histogram_for_model(self, model, iteration):
        if not self._should_log_tensorboard():
            return

        for name, param in model.named_parameters():
            np_param = param.clone().cpu().data.numpy()
            self.summary_writer.add_histogram(name, np_param, iteration)
