import os
import sys
import math
import json
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

class MovingAverage():
    """ Keeps an average window of the specified number of items. """

    def __init__(self, max_window_size=1000):
        self.max_window_size = max_window_size
        self.reset()

    def add(self, elem):
        """ Adds an element to the window, removing the earliest element if necessary. """
        if not math.isfinite(elem):
            print('Warning: Moving average ignored a value of %f' % elem)
            return
        
        self.window.append(elem)
        self.sum += elem

        if len(self.window) > self.max_window_size:
            self.sum -= self.window.popleft()
    
    def append(self, elem):
        """ Same as add just more pythonic. """
        self.add(elem)

    def reset(self):
        """ Resets the MovingAverage to its initial state. """
        self.window = deque()
        self.sum = 0

    def get_avg(self):
        """ Returns the average of the elements in the window. """
        return self.sum / max(len(self.window), 1)

    def __str__(self):
        return str(self.get_avg())
    
    def __repr__(self):
        return repr(self.get_avg())
    
    def __len__(self):
        return len(self.window)


class LogVisualizer():
    def __init__(self, log_file):
        self.log_file = log_file
        print('eLogger {}'.format(self.log_file))

        self.green  = 'tab:green'
        self.blue   = 'tab:blue'
        self.orange = 'tab:orange'
        self.red    = 'tab:red'
        
        self.style_plot = 'seaborn-deep'
        plt.style.use(self.style_plot)
        plt.rcParams.update({'font.size': 22})

        # params
        self.iter_filter  = False  # validations
        self.weight       = 0.6

        self.epoch_filter = True

    def smoothing(self, scalars, weight):
        last = scalars[0]  # First value in the plot (first timestep)
        smoothed = list()
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
            smoothed.append(smoothed_val)                        # Save it
            last = smoothed_val                                  # Anchor the last smoothed value

        return smoothed

    def smoother(self, y, interval=100):
        avg = MovingAverage(interval)

        for i in range(len(y)):
            avg.append(y[i])
            y[i] = avg.get_avg()
        
        return y

    def parse_training(self):
        self.B = []; self.M = []; self.C = []; self.S = []; self.T = []
        self.epoch = []
        
        # open file
        f  = open(self.log_file, "r")

        # read file        
        for line in f.readlines():
            # conver a string dictionary to dictionary
            data = json.loads(line)

            if data['type'] == 'train':
                sub_data = data['data']
                loss     = data['data']['loss']

                self.epoch.append(sub_data['epoch'])
                self.B.append(loss['B'])
                self.M.append(loss['M'])
                self.C.append(loss['C'])
                self.S.append(loss['S'])
                self.T.append(loss['T'])

        # preprocess data
        if self.epoch_filter:
            epochs_keys = list(dict.fromkeys(self.epoch))
            epoch  = np.array(self.epoch)
            B      = np.array(self.B)
            M      = np.array(self.M)
            C      = np.array(self.C)
            S      = np.array(self.S)
            T      = np.array(self.T)

            self.B = []; self.M = []; self.C = []; self.S = []; self.T = []
            self.epoch = []

            for i in epochs_keys:
                index = np.where(epoch==i)
            
                self.epoch.append( np.mean(epoch[index]) )
                self.B.append( np.mean(B[index]) )
                self.M.append( np.mean(M[index]) )
                self.C.append( np.mean(C[index]) )
                self.S.append( np.mean(S[index]) )
                self.T.append( np.mean(T[index]) )

    def parse_validation(self):
        self.epoch = []; self.box_mAP = []; self.mask_mAP = []
        cur_iter   = 0
        prev_iter  = 10

        # open file
        f  = open(self.log_file, "r")

        # read file        
        for line in f.readlines():
            # conver a string dictionary to dictionary
            data = json.loads(line)

            if data['type'] == 'val':
                sub_data   = data['data']

                if self.iter_filter:
                    iterations = sub_data['iter'] 
                    cur_iter   = iterations % 10000

                    if iterations >= 10000 and cur_iter <= prev_iter:
                        self.epoch.append   (sub_data['epoch'])
                        self.box_mAP.append (sub_data['box']['all'])
                        self.mask_mAP.append(sub_data['mask']['all'])
                    prev_iter = cur_iter
                
                else:
                    self.epoch.append  (sub_data['epoch'])
                    self.box_mAP.append (sub_data['box']['all'])
                    self.mask_mAP.append(sub_data['mask']['all'])
                    
    def init_figure(self, ylabel):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,8))
        ax.grid(linestyle=':')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        return fig, ax

    def plot_training(self):
        self.parse_training()

        # # smoothing
        # self.B  = self.smoothing(self.B,  self.weight)
        # self.M  = self.smoothing(self.M,  self.weight)
        # self.C  = self.smoothing(self.C,  self.weight)
        # self.S  = self.smoothing(self.S,  self.weight)
        # self.T  = self.smoothing(self.T,  self.weight)

        label  = ['BBox Loss', 'Mask Loss', 'Conf Loss', 'Segmentation Loss']
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

        fig, ax = self.init_figure(ylabel='Error')
        for idx, val in enumerate([ self.B, self.M, self.C, self.S]):
            # fig, ax = self.init_figure(ylabel='Error')
            ax.plot(self.epoch, self.smoother(val), label=label[idx],  color=colors[idx])
            ax.legend()

        title = '{} Training Loss'.format(self.log_file.split('/')[1])
        ax.set_title(title)

    def plot_validation(self):
        self.parse_validation()

        # smoothing
        # self.box_mAP  = self.smoothing(self.box_mAP,  self.weight)
        # self.mask_mAP = self.smoothing(self.mask_mAP, self.weight)

        fig1, ax1 = self.init_figure(ylabel='mAP')
        # fig2, ax2 = self.init_figure(ylabel='mAP')

        ax1.plot(self.epoch, self.smoother(self.box_mAP),  label='BBox mAP', color=self.red)
        ax1.plot(self.epoch, self.smoother(self.mask_mAP), label='Mask mAP', color=self.green)

        title = '{} Validation mAP'.format(self.log_file.split('/')[1])
        ax1.set_title(title)

        ax1.legend()
        # ax2.legend()

if __name__ == '__main__':
    if len(sys.argv) < 1+1:
        print('Usage: python utils/elogger.py <LOG_FILE>')
        exit()

    vis = LogVisualizer(sys.argv[1])
    vis.plot_validation()
    vis.plot_training()

    plt.show(block=False)
    input('Close all ...')
    plt.close('all')