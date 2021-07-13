import tensorflow as tf
from utils.utilities import *
from math import exp
from numpy import *
import sys
import numpy as np

# 1st Ver.:
# transplanted from 'pytorch_ssim' to TensorFlow.
# 2nd Ver.:
# Referring 'github.com/Momom52/tensorflow-ssim/blob/master/SSIM.py'
# & 'stackoverflow.com/questions/39051451/ssim-ms-ssim-for-tensorflow'
# & TMQI original paper & code.


'''SSIM could also be implemented via
skimage.measure.compare_ssim()/skimage.metrics.structural_similarity()(for ver.>0.18)
or tf.image.ssim()(for higher ver.)'''


'''def gaussian(window_size=11, sigma=1.5):
    gauss = tf.constant([exp(-(x - window_size/2) ** 2 / 2 * sigma ** 2) for x in range(window_size)])
    return gauss / tf.reduce_sum(gauss)

def create_window(window_size=11, sigma=1.5):
    _1D_window = gaussian(window_size=window_size, sigma=sigma)
    _1D_window = tf.expand_dims(_1D_window, axis=1)
    _2D_window = tf.matmul(_1D_window, tf.transpose(_1D_window))
    _2D_window = tf.expand_dims(_2D_window, axis=2)
    _2D_window = tf.expand_dims(_2D_window, axis=3)
    return _2D_window'''

def create_window(size=11, sigma=1.5):
    # Function to mimic the 'fspecial' gaussian MATLAB function

    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / tf.reduce_sum(g)


def ssim_gray(img1, img2, window, k1=0.01, k2=0.03, data_range=1, size_average=True, tmqi=False, sf=16):
    # NOTE that arg. dara_range only works for tmqi=False,
    # there're different data pre-processings in father func. ssim() when tmqi=True

    mu1 = tf.nn.conv2d(img1, window, strides=[1, 1, 1, 1], padding='VALID')
    mu2 = tf.nn.conv2d(img2, window, strides=[1, 1, 1, 1], padding='VALID')

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = tf.nn.conv2d(img1 * img1, window, strides=[1, 1, 1, 1], padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2 * img2, window, strides=[1, 1, 1, 1], padding='VALID') - mu2_sq
    sigma12 = tf.nn.conv2d(img1 * img2, window, strides=[1, 1, 1, 1], padding='VALID') - mu1_mu2

    if tmqi:
        c1, c2 = 0.01, 10
        sigma1 = tf.maximum(1e-6, sigma1_sq) ** 0.5
        sigma2 = tf.maximum(1e-6, sigma2_sq) ** 0.5
        cs_map = (sigma12 + c2) / (sigma1 * sigma2 + c2)
        '''original one, cdf'''
        # sf = 16
        # csf = 100 * 2.6 * (0.0192 + 0.114 * sf) * exp(-(0.114 * sf) ** 1.1)  # Mannos CSF func. about 69.075 when sf=16
        # u = 128 / (1.4 * csf)  # about 1.3236 when sf=16
        # sig = u / 3  # [2.3, 4], 3 the best, about 0.4412 when sf=16
        # sigma1p = tf_normcdf(sigma1, u, sig)  # try to mimic normcdf in MATLAB
        # sigma2p = tf_normcdf(sigma1, u, sig)  # and find it hard to implement in tf.
        '''mimicking using sigmiod func., may cause loss NaN (maybe cuz sigmoid's intrinsic properties?)'''
        # so we use a shifted sigmoid function '1/(1+exp(-x+u)), u=mean=1.3236' as a approximation of
        # the non-linear function (cdf of a normal distribution (u, sig)) in TMQI's SSIM.
        # sigma1p = 1 / (1 + 144 ** (-(sigma1_sq ** 0.5) + u))
        # sigma2p = 1 / (1 + 144 ** (-(sigma2_sq ** 0.5) + u))
        '''mimicking using linear mapping, runs OK but will cause computational error'''
        sigma1p, sigma2p = norm_to_0_1_tf(sigma1), norm_to_0_1_tf(sigma2)

        ssim_map = ((2 * sigma1p * sigma2p + c1) / (sigma1p ** 2 + sigma2p ** 2 + c1)) * cs_map

    else:
        c1 = (k1 * data_range) ** 2
        c2 = (k2 * data_range) ** 2
        cs_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
        ssim_map = ((2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map

    if size_average:
        return tf.reduce_mean(ssim_map), tf.reduce_mean(cs_map)
    else:
        return ssim_map, cs_map


def ssim(img1, img2, data_range=1, size_average=True, channel_weight='AVERAGE', tmqi=False):
    # SSIM in TMQI is actually in a multi-scale way, plz keep tmqi=False & use it in ms_ssim()
    # if tmqi=True is used, it's exactly no longer the same with original TMQI
    window = create_window()

    channels = img1.shape[3]
    ssim_all_channel = []
    for i in range(channels):
        img1_channel = tf.slice(img1, [0, 0, 0, i], [-1, -1, -1, 1])
        img2_channel = tf.slice(img2, [0, 0, 0, i], [-1, -1, -1, 1])
        ssim_per_channel, _ = ssim_gray(img1_channel, img2_channel, window, data_range=data_range, size_average=size_average, tmqi=tmqi)
        ssim_all_channel.append(ssim_per_channel)
    ssim_all_channel = tf.stack(ssim_all_channel, axis=0)  # tf.stack to stack tensors (with same shape) in a list to a whole tensor
    if channel_weight == 'AVERAGE':
        return tf.reduce_mean(ssim_all_channel, axis=0)
    else:
        sys.exit('Only channel_weight==AVERAGE is supported yet.')


def ms_ssim_gray(img1, img2, win, data_range=1, weights=None, size_average=True, tmqi=False, sf=32):
    # Arg. sf only works when tmqi=True

    if weights is None:
        weights = tf.constant([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=tf.float32)
    levels = weights.shape[0]
    msssim = []
    mcs = []

    downsample_filter = tf.ones([2, 2]) / 4
    downsample_filter = tf.expand_dims(downsample_filter, axis=2)
    downsample_filter = tf.expand_dims(downsample_filter, axis=3)

    for i in range(levels):
        sf /= 2

        ssim_map_or_val, cs_map_or_val = ssim_gray(img1, img2, window=win, data_range=data_range, size_average=size_average, tmqi=tmqi, sf=sf)
        msssim.append(ssim_map_or_val ** weights[i])
        mcs.append(cs_map_or_val ** weights[i])

        img1 = tf.nn.conv2d(img1, downsample_filter, strides=[1, 2, 2, 1], padding='SAME')
        img2 = tf.nn.conv2d(img2, downsample_filter, strides=[1, 2, 2, 1], padding='SAME')
        # equal to usage below:
        # img1 = tf.nn.avg_pool(img1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # img2 = tf.nn.avg_pool(img2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # msssim_map = tf.stack(msssim, axis=0)
    # cs_map = tf.stack(mcs, axis=0) # can't stack tensor with different shape

    if size_average:
        return tf.reduce_mean(msssim, axis=0), tf.reduce_mean(mcs, axis=0)
    else:
        # return msssim, mcs # weighted mssim, mcs map with different hw
        sys.exit('usage size_average!=Ture is not allowed in ms-ssim')


def ms_ssim(img1, img2, win_size=11, data_range=1, size_average=True, channel_weight='AVERAGE', tmqi=False):
    # ssim() don't have arg. tmqi, but ms_ssim() do, cuz SSIM in TMQI is in a multi-scale way
    # if tmqi:
        # img1 *= 65535
        # img2 *= 255
        # tf. only allow float16/32, and
        # scaling hdr img to (2 ** 32 - 1) int64 is not allowed, while float32 will overflow its range,
        # so we scaled it to (2 ** 16 -1), different from that in TMQI's MATLAB original code (as above)
        # impact on this difference is not further studied
        # img1 = tf.cast(img1 * 4360503295, dtype=tf.float64)  # 32bit hdr
        # img2 = tf.cast(img2 * 255, dtype=tf.float64)  # 8bit sdr

    win = create_window(win_size)

    channel = img1.shape[3]
    ms_ssim_all_channel = []
    for i in range(channel):
        img1_channel = tf.slice(img1, [0, 0, 0, i], [-1, -1, -1, 1])
        img2_channel = tf.slice(img2, [0, 0, 0, i], [-1, -1, -1, 1])
        ms_ssim_per_channel, _ = ms_ssim_gray(img1_channel, img2_channel, win, data_range, weights=None, size_average=size_average, tmqi=tmqi)
        ms_ssim_all_channel.append(ms_ssim_per_channel)
    ms_ssim_all_channel = tf.stack(ms_ssim_all_channel, axis=0)
    if channel_weight == 'AVERAGE':
        return tf.reduce_mean(ms_ssim_all_channel)
    else:
        sys.exit('Only channel_weight==AVERAGE is supported yet.')
