import logging
import os
from abc import abstractmethod

import torch
from numpy import inf
from os.path import join as ospj
from src.logger import TensorboardWriter
from src.utils.utils import prepare_device

try:
    import wandb
except:
    pass
class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, config, log_dir):

        self.config = config # General config (model arch, optimizer, lr_scheduler, etc.)
        self.trainer_config = self.config['trainer_config'] # Training details such as num_epochs etc.

        # Setup a logger (just for cleaner log files)
        self._configure_logging(log_dir)

        # Read and save some of the configs
        self.epochs = self.trainer_config['epochs']
        self.save_period = self.trainer_config['save_period']
        self.eval_period = self.trainer_config['eval_period']

        # Configure how to monitor training and how to make checkpoints
        self._configure_monitoring()

        # Setup the checkpoint directory (where to save checkpoints)
        self.checkpoint_dir = ospj(
            self.trainer_config['save_dir'], self.config['name']
        )
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        else:
            print(f'Warning! Save dir {self.checkpoint_dir} already exists!'+\
                'Existing checkpoints will be overwritten!')
        

        # Setup visualization writer instance (Tensorboard, WandB)
        self.writer = None
        if self.trainer_config['tensorboard']:
            self.writer = TensorboardWriter(ospj(log_dir, 'tensorboard'), self.logger)
        self.wandb_enabled = self.trainer_config['wandb']

        # Prepare for (multi-device) GPU training
        # This part doesn't do anything if you don't have a GPU.
        self._device, self._device_ids = prepare_device(self.trainer_config['n_gpu'])
        
        self.start_epoch = 1
        self.best_epoch = 1
        self.current_epoch = 1

    def _configure_logging(self, log_dir):
        self.logger = logging.getLogger()
        self.logger.setLevel(1)
        formatter = logging.Formatter('%(asctime)s:%(levelname)s : %(message)s')
        if not os.path.exists(ospj(log_dir)):
            os.mkdir(log_dir)
        _log_file = ospj(log_dir, self.config['name']+".log")
        if os.path.exists(_log_file):
            print(f'Warning! Log file {_log_file} already exists! The logs will be appended!')
        file_handler = logging.FileHandler(_log_file)
        file_handler.setFormatter(formatter)
        if (self.logger.hasHandlers()):
            self.logger.handlers.clear()
        self.logger.addHandler(file_handler)
    
    def _configure_monitoring(self):
        self.monitor = self.trainer_config.get('monitor', 'off')
        if self.monitor == 'off':
            self.monitor_mode = 'off'
            self.monitor_best = 0
        else:
            self.monitor_mode, self.monitor_metric = self.monitor.split()
            assert self.monitor_mode in ['min', 'max']

            self.monitor_best = inf if self.monitor_mode == 'min' else -inf

            # Only enable early stopping if given and above 0
            self.early_stop = self.trainer_config.get('early_stop', inf)
            if self.early_stop <= 0:
                self.early_stop = inf # training proceeds till the very last epoch

    @abstractmethod # To be implemented by the child classes!
    def _train_epoch(self):
        """
        Training logic for an epoch. Only takes care of doing a single training loop.

        :return: A dict that contains average loss and metric(s) information in this epoch.
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        self.logger.info("------------ New Training Session ------------")
        self.not_improved_count = 0        
        if self.wandb_enabled: wandb.watch(self.model, self.criterion, log='all')

        for epoch in range(self.start_epoch, self.epochs + 1):
            self.current_epoch = epoch
            train_result = self._train_epoch()

            # save logged informations into log dict
            log = {'epoch': self.current_epoch}
            log.update(train_result)

            if self.should_evaluate():
                eval_result = self.evaluate()
                # save eval information to the log dict as well
                log.update({f'eval_{key}': value for key, value in eval_result.items()})    

            if self.monitor_mode != 'off' : # Then there is a metric to monitor
                if self.monitor_metric in log: # Then we have measured it in this epoch
                    #################################################################################################
                    # TODO: Q2.b: Use the dictionaries above to see if this is the best epoch based                 #
                    # on self.monitor_metric. If so, use self.save_model() to save the best model checkpoint.       #
                    # Don't forget to pre-pend self.checkpoint_dir to the path argument.                            #
                    # We also recommend printing the epoch number so that later from the logs.                      #
                    # check whether model performance improved or not, according to specified metric(monitor_metric)#
                    # These were configured in the self._configure_monitoring()                                     #
                    # You may have to check for min or max value based on monitor_mode                              #
                    # (e.g for loss values we would want min, but for accuracy we want max.)                        #
                    #################################################################################################   

                    pass



                    ############################################################################################
                    # TODO: Q2.c: Based self.monitor_metric and whether we have had improvements in            #
                    # the last self.early_stop steps, see if you should break the training loop.               #
                    ############################################################################################



                    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
                else:
                    ## The metric wasn't measured in this epoch. Don't change not_impoved_count or similar things here!!!
                    self.logger.warning(f"Warning: At epoch {self.current_epoch} Metric '{self.monitor_metric}'"+\
                                " wasn't measured. Not monitoring it for this epoch.")
            
            # print logged information to the screen
            for key, value in log.items():
                self.logger.info(f'    {key[:15]}: {value:0.5f}')

            if self.wandb_enabled: wandb.log(log)

            if self.current_epoch % self.save_period == 0:
                # Just to regularly save the model every save_period epochs
                path = os.path.join(self.checkpoint_dir, f'E{self.current_epoch}_model.pth')
                self.save_model(path=path)

        # Always save the last model
        path = os.path.join(self.checkpoint_dir, f'last_model.pth')
        self.save_model(path=path)

    def should_evaluate(self):
        """
        Based on the self.current_epoch and self.eval_interval, determine if we should evaluate.
        You can take hint from saving logic implemented in BaseTrainer.train() method

        returns a Boolean
        """
        ###  TODO  ################################################
        # Based on the self.current_epoch and self.eval_interval, determine if we should evaluate.
        # You can take hint from saving logic implemented in BaseTrainer.train() method
        return True
        #########################################################
    
    @abstractmethod # To be implemented by the child classes!
    def evaluate(self, loader=None):
        """
        Evaluate the model on the val_loader given at initialization

        :param loader: A Dataloader to be used for evaluation. If not given, it will use the 
        self._eval_loader that's set during initialization..
        :return: A dict that contains metric(s) information for validation set
        """
        raise NotImplementedError
    
    def save_model(self, path=None):
        """
        Saves only the model parameters.
        : param path: path to save model (including filename.)
        """
        self.logger.info("Saving checkpoint: {} ...".format(path))
        torch.save(self.model.state_dict(), path)
        self.logger.info("Checkpoint saved.")
    
    def load_model(self, path=None):
        """
        Loads model params from the given path.
        : param path: path to save model (including filename.)
        """
        self.logger.info("Loading checkpoint: {} ...".format(path))
        self.model.load_state_dict(torch.load(path))
        self.logger.info("Checkpoint loaded.")