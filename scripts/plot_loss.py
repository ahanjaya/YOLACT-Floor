import re, sys, os
import matplotlib.pyplot as plt

# from utils.functions import MovingAverage

import torch
import torch.nn as nn
import os
import math
from collections import deque
from pathlib import Path

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


with open(sys.argv[1], 'r') as f:
	inp = f.read()

patterns = {
	'train': re.compile(r'\[\s*(?P<epoch>\d+)\]\s*(?P<iteration>\d+) \|\| B: (?P<b>\S+) \| C: (?P<c>\S+) \| M: (?P<m>\S+) \|( S: (?P<s>\S+) \|)? T: (?P<t>\S+)'),
	'val': re.compile(r'\s*(?P<type>[a-z]+) \|\s*(?P<all>\S+)')
}
data = {key: [] for key in patterns}

for line in inp.split('\n'):
	for key, pattern in patterns.items():
		f = pattern.search(line)
		
		if f is not None:
			datum = f.groupdict()
			for k, v in datum.items():
				if v is not None:
					try:
						v = float(v)
					except ValueError:
						pass
					datum[k] = v
			
			if key == 'val':
				datum = (datum, data['train'][-1])
			data[key].append(datum)
			break


def smoother(y, interval=100):
	avg = MovingAverage(interval)

	for i in range(len(y)):
		avg.append(y[i])
		y[i] = avg.get_avg()
	
	return y

def plot_train(data):
	plt.title(os.path.basename(sys.argv[1]) + ' Training Loss')
	plt.xlabel('Iteration')
	plt.ylabel('Loss')

	loss_names = ['BBox Loss', 'Conf Loss', 'Mask Loss']

	x = [x['iteration'] for x in data]
	plt.plot(x, smoother([y['b'] for y in data]))
	plt.plot(x, smoother([y['c'] for y in data]))
	plt.plot(x, smoother([y['m'] for y in data]))

	if data[0]['s'] is not None:
		plt.plot(x, smoother([y['s'] for y in data]))
		loss_names.append('Segmentation Loss')

	plt.legend(loss_names)
	plt.show()

def plot_val(data):
	plt.title(os.path.basename(sys.argv[1]) + ' Validation mAP')
	plt.xlabel('Epoch')
	plt.ylabel('mAP')

	x = [x[1]['epoch'] for x in data if x[0]['type'] == 'box']
	plt.plot(x, [x[0]['all'] for x in data if x[0]['type'] == 'box'])
	plt.plot(x, [x[0]['all'] for x in data if x[0]['type'] == 'mask'])

	plt.legend(['BBox mAP', 'Mask mAP'])
	plt.show()

if len(sys.argv) > 2 and sys.argv[2] == 'val':
	plot_val(data['val'])
else:
	plot_train(data['train'])
