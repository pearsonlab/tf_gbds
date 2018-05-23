import errno
import numpy as np
# import scipy
from scipy.io import loadmat
import math
from scipy.stats import norm
# import pandas as pd

import sys
import os
from os.path import join
curr_dir = os.getcwd()
# sys.path.append(join(curr_dir, 'common_lib'))
sys.path.append(join(curr_dir, 'util'))

from math_related import *

minScreen = np.zeros((1, 1))
maxScreen = np.array([1920, 1080])
rnd_seed = np.random.seed(1234)
data_ratio = 0.8
smooth_sigma = 2
num_prey = 1

# Coordinate related column of the data
self_coord = np.arange(2)
if num_prey == 1:
    # subj x and y, prey x and y, gaze x and y
    prey_coord = np.array([2, 3])
    eye_coord = np.array([4, 5])
    value_col = np.array([7])
else:
    # subj x and y, prey1 x and y, prey2 x and y, gaze x and y
    prey_coord = np.linspace(2, 5, 4).astype('int')
    eye_coord = np.array([5, 6])
    value_col = np.array([8, 9])

def check_path(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def load_cost(file_loc, file_name):
    """
    load data from indicated location and by name.
    """
    cost_map = loadmat(file_loc+file_name)
    cost_map = cost_map['cost_grid']

    return cost_map

def load_data(file_loc, file_name):
    """
    load data from indicated location and by name.
    """
    raw_data = loadmat(file_loc+file_name)
    raw_data = raw_data['data']
    raw_data = np.transpose(raw_data[0]) # Transpose for having rows to be longer

    return raw_data

def smooth_trial(trial, sigma_val):
    rtrial = trial.copy()

    if sigma_val is None:
        sigma_val = 4 # Default sigma

    for i in range(rtrial.shape[1]):
        rtrial[:, i] = gauss_convolve(rtrial[:, i], sigma_val, extrapolate=True)
    return rtrial

def gauss_convolve(x, sigma, extrapolate=False):
    edge = int(math.ceil(5 * sigma))
    fltr = norm.pdf(range(-edge, edge), loc=0, scale=sigma)
    fltr = fltr / sum(fltr)

    buff = np.ones((1, edge))[0]

    szx = x.size

    xx = np.append((buff * x[0]), x)

    if extrapolate:
        # linear extrapolation for end edge buffer
        end_dx = x[-1] - x[-2]
        end_buff = np.cumsum(end_dx * np.ones(edge)) + x[-1]
    else:
        # pad with last value
        end_buff = buff * x[-1]

    xx = np.append(xx, end_buff)

    y = np.convolve(xx, fltr, mode='valid')
    y = y[:szx]
    return y

def normalize_seq_data(data, coords):
    min_val = minScreen[0]
    norm_data = []
    smoothed = []
    for iN in range(0, np.size(data, 0)):
        for iDim in range(0, len(coords)):
            in_data = data[iN][:, [coords[iDim]]]

            if iDim == 0:
                max_val = maxScreen[0]
            else:
                max_val = maxScreen[1]

            norm_coord = (2 * (in_data - min_val) / (max_val - min_val)) - 1
            if iDim == 0:
                stack_d = norm_coord.reshape((-1, 1))
            else:
                stack_d = np.hstack((stack_d, norm_coord.reshape((-1, 1))))
        smoothed.append(smooth_trial(stack_d.astype(np.float32), smooth_sigma))
        norm_data.append(stack_d.astype(np.float32))

    return norm_data, smoothed

def normalize_eye_data(data):
    min_val = minScreen

    norm_eye_data = []
    for iN in range(0, np.size(data, 0)):
        for iDim in range(0, len(eye_coord)):
            in_data = data[iN][:, [eye_coord[iDim]]]

            if iDim == 0:
                max_val = maxScreen[0]
            else:
                max_val = maxScreen[1]

            norm_coord = (2 * (in_data - min_val) / (max_val - min_val)) - 1
            if iDim == 0:
                stack_d = norm_coord.reshape((-1, 1))
            else:
                stack_d = np.hstack((stack_d, norm_coord.reshape((-1, 1))))
        norm_eye_data.append(smooth_trial(stack_d.astype(np.float32), smooth_sigma))

    return norm_eye_data

def divide_data(data, val):
    """
    This function divides normalized data into training and validation.
    This gets raw data since the normalized value information is within the raw data
    """
    rnd_seed = np.random.seed(1234)
    tr_data = []
    vd_data = []
    tr_val_data = []
    vd_val_data = []
    for iRnd in range(0, np.size(data, 0)):
        if np.random.uniform() < data_ratio:
            tr_data.append(data[iRnd])
            tr_val_data.append(np.squeeze(val[iRnd][:, [value_col]].astype(np.float32), axis=1))
        else:
            vd_data.append(data[iRnd])
            vd_val_data.append(np.squeeze(val[iRnd][:, [value_col]].astype(np.float32), axis=1))

    return tr_data, vd_data, tr_val_data, vd_val_data
