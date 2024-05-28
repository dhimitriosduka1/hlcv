import os
import torch
import torch.nn as nn

from tqdm import tqdm

from torchvision.utils import make_grid
from .base_trainer import BaseTrainer
from src.utils.utils import MetricTracker

class VGGTrainer(BaseTrainer):

    def __init__(self, config, log_dir, train_loader, eval_loader=None):
        """
        Create the model, loss criterion, optimizer, and dataloaders
        And anything else that might be needed during training. (e.g. device type)
        """
        super().__init__(config, log_dir)   

        self.model = self.config['model_arch'](**self.config['model_args'])
        self.model.to(self._device)
        if len(self._device_ids) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self._device_ids)

        # Simply Log the model (enable if you want to see the model architecture)
        # self.logger.info(self.model)

        # Prepare Losses
        self.criterion = self.config['criterion'](**self.config['criterion_args'])

        # Prepare Optimizer
        trainable_params = self.model.parameters()

        # Configure the optimizer and lr scheduler
        # These are usually Python Partial() objects that have all the options already inserted.
        self.optimizer = self.config['optimizer'](trainable_params)
        self.lr_scheduler = self.config['lr_scheduler'](self.optimizer) 


        # Set DataLoaders
        self._train_loader = train_loader
        self._eval_loader = eval_loader
        
        self.log_step = self.trainer_config['log_step']

        # Prepare Metrics
        self.metric_functions = self.config['metrics']
        self.train_metrics = MetricTracker(
            keys=['loss'] + [metric_key for metric_key in self.metric_functions.keys()],
            writer=self.writer)
        self.eval_metrics = MetricTracker(
            keys=['loss'] + [metric_key for metric_key in self.metric_functions.keys()],
            writer=self.writer)

    def _train_epoch(self):
        """
        Training logic for an epoch. Only takes care of doing a single training loop.

        :return: A dict that contains average loss and metric(s) information in this epoch.
        """
        #######
        # Set model to train mode
        ######
        self.model.train()
        self.train_metrics.reset()

        self.logger.debug(f"==> Start Training Epoch {self.current_epoch}/{self.epochs}, lr={self.optimizer.param_groups[0]['lr']:.6f} ")

        pbar = tqdm(total=len(self._train_loader) * self._train_loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        for batch_idx, (images, labels) in enumerate(self._train_loader):

            images = images.to(self._device)
            labels = labels.to(self._device)


            output = self.model(images)
            loss = self.criterion(output, labels)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.writer is not None:
                self.writer.set_step((self.current_epoch - 1) * len(self._train_loader) + batch_idx)
            
            # Update all the train_metrics with new values.
            self.train_metrics.update('loss', loss.item())
            for metric_key, metric_func in self.metric_functions.items():
                self.train_metrics.update(metric_key, metric_func.compute(output, labels))

            pbar.set_description(f"Train Epoch: {self.current_epoch} Loss: {loss.item():.4f}")

            # This is just a sample to show how you can log things to your writer.
            # if self.writer is not None and batch_idx % self.log_step == 0:
            #     self.writer.add_image('input_train', make_grid(images.cpu(), nrow=8, normalize=True))

            pbar.update(self._train_loader.batch_size)

        log_dict = self.train_metrics.result()
        pbar.close()
        self.lr_scheduler.step()

        self.logger.debug(f"==> Finished Epoch {self.current_epoch}/{self.epochs}.")
        
        return log_dict
    
    @torch.no_grad()
    def evaluate(self, loader=None):
        """
        Evaluate the model on the val_loader given at initialization

        :param loader: A Dataloader to be used for evaluatation. If not given, it will use the 
        self._eval_loader that's set during initialization..
        :return: A dict that contains metric(s) information for validation set
        """
        if loader is None:
            assert self._eval_loader is not None, 'loader was not given and self._eval_loader not set either!'
            loader = self._eval_loader

        self.model.eval()
        self.eval_metrics.reset()

        self.logger.debug(f"++> Evaluate at epoch {self.current_epoch} ...")

        pbar = tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        for batch_idx, (images, labels) in enumerate(loader):
            
            images = images.to(self._device)
            labels = labels.to(self._device)

            output = self.model(images)
            loss = self.criterion(output, labels)

            if self.writer is not None: self.writer.set_step((self.current_epoch - 1) * len(loader) + batch_idx, 'valid')
            self.eval_metrics.update('loss', loss.item())
            for metric_key, metric_func in self.metric_functions.items():
                self.eval_metrics.update(metric_key, metric_func.compute(output, labels))

            pbar.set_description(f"Eval Loss: {loss.item():.4f}")
            pbar.update(loader.batch_size)

            # This is just a sample to show how you can log things to your writer.
            # if self.writer is not None:
            #     self.writer.add_image('input_valid', make_grid(images.cpu(), nrow=8, normalize=True))

            
        # add histogram of model parameters to the tensorboard (This can be very slow for big models.)
        # if self.writer is not None:
        #     for name, p in self.model.named_parameters():
        #         self.writer.add_histogram(name, p, bins='auto')

        pbar.close()
        self.logger.debug(f"++> Evaluate epoch {self.current_epoch} Finished.")
        
        return self.eval_metrics.result()