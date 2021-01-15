# Copyright (c) Facebook, Inc. and its affiliates.

import unittest
from unittest.mock import MagicMock

import torch
from mmf.common.report import Report
from mmf.trainers.lightning_trainer import LightningTrainer
from mmf.utils.build import build_optimizer
from mmf.utils.general import clip_gradients
from omegaconf import OmegaConf
from pytorch_lightning.callbacks.base import Callback
from tests.test_utils import NumbersDataset, SimpleLightningModel, SimpleModel
from tests.trainers.test_training_loop import TrainerTrainingLoopMock


class LightningTrainerMock(LightningTrainer):
    def __init__(
        self,
        config,
        max_updates,
        max_epochs=None,
        callback=None,
        num_data_size=100,
        batch_size=1,
        update_frequency=1,
        lr_scheduler=False,
        grad_clipping_config=None,
        fp16=False,
    ):
        self.data_module = MagicMock()
        self._benchmark = False
        self._distributed = False
        self._gpus = None
        self._gradient_clip_val = False
        self._num_nodes = 1
        self._deterministic = True
        self._automatic_optimization = False
        self._callbacks = None
        if callback:
            self._callbacks = [callback]
        self.config = config
        dataset = NumbersDataset(num_data_size)
        self.data_module.train_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1,
            drop_last=False,
        )
        self.data_module.train_loader.current_dataset = MagicMock(return_value=dataset)
        self.training_config = self.config.training
        self.training_config.update_frequency = update_frequency
        self.training_config.fp16 = fp16
        self.training_config.max_updates = max_updates
        self.training_config.max_epochs = max_epochs
        self.training_config.lr_scheduler = lr_scheduler
        if grad_clipping_config:
            self.training_config.clip_gradients = True
            self.training_config.max_grad_l2_norm = grad_clipping_config[
                "max_grad_l2_norm"
            ]
            self.training_config.clip_norm_mode = grad_clipping_config["clip_norm_mode"]
        else:
            self.training_config.clip_gradients = False


class TestLightningTrainer(unittest.TestCase, Callback):
    @classmethod
    def setup_class(self):
        self.lightning_losses = []
        self.mmf_losses = []
        self.mmf_grads = []
        self.lightning_grads = []

        # tests that require callbacks
        self._test_same_loss_computation = False
        self._test_grad_clipping = False
        self._test_grad_clipping_compared_to_mmf_is_same = False

        self.grad_clip_magnitude = 0.15
        self.grad_clipping_config = {
            "max_grad_l2_norm": self.grad_clip_magnitude,
            "clip_norm_mode": "all",
        }

    def get_trainer_config(self):
        return OmegaConf.create(
            {
                "distributed": {},
                "run_type": "train",
                "training": {
                    "detect_anomaly": False,
                    "evaluation_interval": 4,
                    "log_interval": 2,
                    "update_frequency": 1,
                    "fp16": False,
                    "lr_scheduler": False,
                },
                "optimizer": {"type": "adam_w", "params": {"lr": 5e-5, "eps": 1e-8}},
                "scheduler": {
                    "type": "warmup_linear",
                    "params": {"num_warmup_steps": 8, "num_training_steps": 8},
                },
            }
        )

    def get_lightning_trainer(
        self,
        max_updates,
        max_epochs=None,
        batch_size=1,
        model_size=1,
        update_frequency=1,
        callback=None,
        lr_scheduler=False,
        grad_clipping_config=None,
        fp16=False,
    ):
        torch.random.manual_seed(2)
        trainer = LightningTrainerMock(
            config=self.get_trainer_config(),
            max_updates=max_updates,
            max_epochs=max_epochs,
            callback=callback,
            batch_size=batch_size,
            update_frequency=update_frequency,
            lr_scheduler=lr_scheduler,
            grad_clipping_config=grad_clipping_config,
            fp16=fp16,
        )
        trainer.model = SimpleLightningModel(model_size, config=trainer.config)
        trainer.model.train()
        return trainer

    def get_mmf_trainer(
        self,
        model_size=1,
        num_data_size=100,
        max_updates=5,
        max_epochs=None,
        on_update_end_fn=None,
        fp16=False,
        scheduler_config=None,
        grad_clipping_config=None,
    ):
        torch.random.manual_seed(2)
        model = SimpleModel(model_size)
        model.train()
        trainer_config = self.get_trainer_config()
        optimizer = build_optimizer(model, trainer_config)
        trainer = TrainerTrainingLoopMock(
            num_data_size,
            max_updates,
            max_epochs,
            config=trainer_config,
            optimizer=optimizer,
            on_update_end_fn=on_update_end_fn,
            fp16=fp16,
            scheduler_config=scheduler_config,
            grad_clipping_config=grad_clipping_config,
        )
        model.to(trainer.device)
        trainer.model = model
        return trainer

    def test_epoch_over_updates(self):
        trainer = self.get_lightning_trainer(max_updates=2, max_epochs=0.04)
        self._prepare_trainer(trainer)
        self.assertEqual(trainer._max_updates, 4)

        self._check_values(trainer, 0, 0)
        trainer.trainer.fit(trainer.model, trainer.data_module.train_loader)
        self._check_values(trainer, 4, 0)

    def test_fractional_epoch(self):
        trainer = self.get_lightning_trainer(max_updates=None, max_epochs=0.04)
        self._prepare_trainer(trainer)
        self.assertEqual(trainer._max_updates, 4)

        self._check_values(trainer, 0, 0)
        trainer.trainer.fit(trainer.model, trainer.data_module.train_loader)
        self._check_values(trainer, 4, 0)

    def test_updates(self):
        trainer = self.get_lightning_trainer(max_updates=2, max_epochs=None)
        self._prepare_trainer(trainer)
        self.assertEqual(trainer._max_updates, 2)

        self._check_values(trainer, 0, 0)
        trainer.trainer.fit(trainer.model, trainer.data_module.train_loader)
        self._check_values(trainer, 2, 0)

    def _check_values(self, trainer, current_iteration, current_epoch):
        self.assertEqual(trainer.trainer.global_step, current_iteration)
        self.assertEqual(trainer.trainer.current_epoch, current_epoch)

    def test_grad_clipping(self):
        # the test in the callback `on_after_backward`
        self._test_grad_clipping = True
        trainer = self.get_lightning_trainer(
            max_updates=10,
            max_epochs=None,
            grad_clipping_config=self.grad_clipping_config,
            callback=self,
        )
        self._prepare_trainer(trainer)
        trainer.trainer.fit(trainer.model, trainer.data_module.train_loader)

    def test_grad_clipping_compared_to_mmf_is_same(self):
        self._test_grad_clipping_compared_to_mmf_is_same = True
        # compute mmf_trainer training model grads
        mmf_trainer = self.get_mmf_trainer(
            max_updates=5,
            max_epochs=None,
            grad_clipping_config=self.grad_clipping_config,
        )

        def _finish_update():
            clip_gradients(
                mmf_trainer.model, mmf_trainer.num_updates, None, mmf_trainer.config
            )
            for param in mmf_trainer.model.parameters():
                mmf_grad = torch.clone(param.grad).detach().item()
                self.mmf_grads.append(mmf_grad)

            mmf_trainer.scaler.step(mmf_trainer.optimizer)
            mmf_trainer.scaler.update()
            mmf_trainer.num_updates += 1

        mmf_trainer._finish_update = _finish_update
        mmf_trainer.training_loop()

        # the test in the callback `on_after_backward`
        trainer = self.get_lightning_trainer(
            max_updates=5,
            max_epochs=None,
            grad_clipping_config=self.grad_clipping_config,
            callback=self,
        )
        self._prepare_trainer(trainer)
        trainer.trainer.fit(trainer.model, trainer.data_module.train_loader)

    def test_lr_schedule_with_or_without_is_different(self):
        # note, be aware some of the logic also is in the SimpleLightningModel
        trainer1 = self.get_lightning_trainer(max_updates=8, lr_scheduler=True)
        self._prepare_trainer(trainer1)
        trainer1.trainer.fit(trainer1.model, trainer1.data_module.train_loader)

        trainer2 = self.get_lightning_trainer(max_updates=8)
        self._prepare_trainer(trainer2)
        trainer2.trainer.fit(trainer2.model, trainer2.data_module.train_loader)

        last_model_param1 = list(trainer1.model.parameters())[-1]
        last_model_param2 = list(trainer2.model.parameters())[-1]
        self.assertFalse(torch.allclose(last_model_param1, last_model_param2))

    def test_lr_schedule_compared_to_mmf_is_same(self):
        trainer_config = self.get_trainer_config()
        mmf_trainer = self.get_mmf_trainer(
            max_updates=8, max_epochs=None, scheduler_config=trainer_config.scheduler
        )
        mmf_trainer.training_loop()

        trainer = self.get_lightning_trainer(max_updates=8, lr_scheduler=True)
        self._prepare_trainer(trainer)
        trainer.trainer.fit(trainer.model, trainer.data_module.train_loader)

        mmf_trainer.model.to(trainer.model.device)
        last_model_param1 = list(mmf_trainer.model.parameters())[-1]
        last_model_param2 = list(trainer.model.parameters())[-1]
        self.assertTrue(torch.allclose(last_model_param1, last_model_param2))

    def test_grad_accumulate(self):
        trainer1 = self.get_lightning_trainer(
            update_frequency=2, callback=self, max_updates=2, batch_size=3
        )
        self._prepare_trainer(trainer1)
        trainer1.trainer.fit(trainer1.model, trainer1.data_module.train_loader)

        trainer2 = self.get_lightning_trainer(
            update_frequency=1, callback=self, max_updates=2, batch_size=6
        )
        self._prepare_trainer(trainer2)
        trainer2.trainer.fit(trainer2.model, trainer2.data_module.train_loader)

        for param1, param2 in zip(
            trainer1.model.parameters(), trainer2.model.parameters()
        ):
            self.assertTrue(torch.allclose(param1, param2))

    def _prepare_trainer(self, trainer):
        trainer.configure_device()
        trainer._calculate_max_updates()
        trainer._calculate_gradient_clip_val()
        trainer._load_trainer()

    def test_same_loss_computation(self):
        # check to see the same losses between the two trainers
        # under the same conditions
        self._test_same_loss_computation = True

        # compute mmf_trainer training losses
        def _on_update_end(report, meter, should_log):
            self.mmf_losses.append(report["losses"]["loss"].item())

        mmf_trainer = self.get_mmf_trainer(
            max_updates=5, max_epochs=None, on_update_end_fn=_on_update_end
        )
        mmf_trainer.training_loop()

        # compute lightning_trainer training losses
        trainer = self.get_lightning_trainer(callback=self, max_updates=5)
        self._prepare_trainer(trainer)
        trainer.trainer.fit(trainer.model, trainer.data_module.train_loader)

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if self._test_same_loss_computation:
            output = outputs[0][0]["extra"]
            report = Report(output["input_batch"], output)
            self.lightning_losses.append(report["losses"]["loss"].item())

    def on_after_backward(self, trainer, pl_module):
        if self._test_grad_clipping:
            for param in pl_module.parameters():
                self.assertLessEqual(param.grad, self.grad_clip_magnitude)

        if self._test_grad_clipping_compared_to_mmf_is_same:
            for lightning_param in pl_module.parameters():
                lightning_grad = torch.clone(lightning_param.grad).detach().item()
                self.lightning_grads.append(lightning_grad)

    def on_train_end(self, trainer, pl_module):
        if self._test_same_loss_computation:
            for lightning_loss, mmf_loss in zip(self.lightning_losses, self.mmf_losses):
                self.assertEqual(lightning_loss, mmf_loss)

        if self._test_grad_clipping_compared_to_mmf_is_same:
            for lightning_grad, mmf_grad in zip(self.lightning_grads, self.mmf_grads):
                self.assertEqual(lightning_grad, mmf_grad)

        self._test_same_loss_computation = False
        self._test_grad_clipping = False
        self._test_grad_clipping_compared_to_mmf_is_same = False
