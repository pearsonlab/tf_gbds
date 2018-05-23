"""
All the math related functions are located in this function
1. Smoothing by Gaussian Convolution is here.

"""

import numpy as np
import math
from scipy.stats import norm
import pandas as pd

def smooth_trial(trial, sigma_val):
    rtrial = trial.copy()

    if sigma_val is None:
        sigma_val = 4

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

def convolution(A, B):
    lengthA = np.size(A)
    lengthB = np.size(B)
    C = np.zeros(lengthA + lengthB - 1)

    for m in np.arange(lengthA):
        for n in np.arange(lengthB):
            C[m + n] = C[m + n] + A[m] * B[n]

    return C

def crosscorrelation(A, B):
    return convolution(np.conj(A), B[::-1])
