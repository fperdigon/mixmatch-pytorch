#============================================================
#
#  Pytorchtools
#  This file contains useful classes for pytorch implementations
#   
#  author: Francisco Perdigon Romero
#  email: fperdigon88@gmail.com
#  github id: fperdigon
#  MedICAL Lab
#
#============================================================

import numpy as np
import math
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, min_delta=0.01, mode='min', verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            mode (str): Defines if min or max ar the best for the metric.
                            Default: min
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.mode = mode #(min or max)
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        if self.mode == 'min':
            self.metric_best = np.Inf
        elif self.mode == 'max':
            self.metric_best = -np.Inf
        else:
            raise NameError('The mode parameter must be an string equal to min or max ')

    def __call__(self, metric_value):

        score = -metric_value

        if self.mode == 'min':
            if self.best_score is None:
                self.best_score = score
                self.metric_best = metric_value
            elif score < self.best_score:
                self.counter += 1
                print('\nEarlyStopping counter: %i out of %i. Last best: %.6f' % (self.counter, self.patience, self.metric_best))
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                if math.fabs(self.best_score-score) > self.min_delta:
                    self.best_score = score
                    self.metric_best = metric_value
                    self.counter = 0

        elif self.mode == 'max':
            if self.best_score is None:
                self.best_score = score
                self.metric_best = metric_value
            elif score > self.best_score:
                self.counter += 1
                print('\nEarlyStopping counter: %i out of %i. Last best: %.6f' % (self.counter, self.patience, self.metric_best))
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                if math.fabs(self.best_score - score) > self.min_delta:
                    self.best_score = score
                    self.metric_best = metric_value
                    self.counter = 0



class ModelCheckpoint:
    """Model checpoint based on the inprovement of an specific metric """

    def __init__(self, checkpoint_fn='checkpoint.pt', mode='min', min_delta=0.01, verbose=False):
        """
        Args:
            checkpoint_fn (str): Name for the model checkpoint save file
                            Default: 'checkpoint.pt'
            mode (str): Defines if min or max ar the best for the metric.
                            Default: min
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """

        self.mode = mode  # (min or max)
        self.min_delta = min_delta
        self.checkpoint_fn = checkpoint_fn
        self.verbose = verbose
        self.best_score = None

        if self.mode == 'min':
            self.metric_best = np.Inf
        elif self.mode == 'max':
            self.metric_best = -np.Inf
        else:
            raise NameError('The mode parameter must be an string equal to min or max ')


    def __call__(self, current_metric, model):

        score = -current_metric

        if self.mode == 'min':
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(current_metric, model)
            elif score < self.best_score:
                pass
            else:
                if math.fabs(self.best_score - score) > self.min_delta:
                    self.best_score = score
                    self.save_checkpoint(current_metric, model)

        elif self.mode == 'max':
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(current_metric, model)
            elif score > self.best_score:
                pass
            else:
                if math.fabs(self.best_score - score) > self.min_delta:
                    self.best_score = score
                    self.save_checkpoint(current_metric, model)



    def save_checkpoint(self, current_metric, model):
        '''Saves model when metric inproves.'''
        if self.verbose:
            print('\nMetric inproved (%.6f --> %.6f).  Saving model '% (self.metric_best, current_metric) + self.checkpoint_fn + ' ...')
        torch.save(model.state_dict(), self.checkpoint_fn)
        self.metric_best = current_metric


class state_variables:
    """
    This class implements a  variable that can be updated and return an average of valies
    up to the current element.
    This reduces the amount of codes in the Pytorch training scheme
    """
    def __init__(self, avg_type='std_avg'):
        self.avg_value = 0
        self.counter = 0
        self.avg_type = avg_type
        self.last_value = None

    def update(self, new_value):
        """
        This function updates the avg_value and the last_value variables using different avg formulas
        For now the only one implemented is the standard mean calculation
        :param new_value:
        :return:
        """
        self.counter += 1
        self.last_value = new_value

        # std_avg calculation
        self.avg_value = (new_value + (self.counter - 1) * self.avg_value) / self.counter

    def get_avg_value(self):
        return self.avg_value

    def get_last_value(self):
        return self.last_value

    def reset(self):
        self.avg_value = 0
        self.counter = 0
        self.last_value = None
