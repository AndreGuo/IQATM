import tensorflow as tf
import numpy as np
from utils.utilities import *


'''
def _img_grad(img):
  """(transplanted from higher ver. of tf.)
  Returns image gradients (dy, dx) for each color channel.

  Both output tensors have the same shape as the input: [batch_size, h, w,
  d]. The gradient values are organized so that [I(x+1, y) - I(x, y)] is in
  location (x, y). That means that dy will always have zeros in the last row,
  and dx will always have zeros in the last column.

  Arguments:
    image: Tensor with shape [batch_size, h, w, d].

  Returns:
    Pair of tensors (dy, dx) holding the vertical and horizontal image
    gradients (1-step finite difference).

  Raises:
    ValueError: If `image` is not a 4D tensor.
  """
  if img.get_shape().ndims != 4:
    raise ValueError('image_gradients expects a 4D tensor '
                     '[batch_size, h, w, d], not %s.', img.get_shape())
  image_shape = tf.shape(img)
  batch_size, height, width, depth = tf.unstack(image_shape)
  # operation below utilized tf's index to replace tf.slice
  dy = img[:, 1:, :, :] - img[:, :-1, :, :]  # (2nd to last) - (1st to last but one) row
  dx = img[:, :, 1:, :] - img[:, :, :-1, :]  # (2nd to last) - (1st to last but one) column
  # equal to 'prewitt' optator (?)

  # Return tensors with same size as original image by concatenating
  # zeros. Place the gradient [I(x+1,y) - I(x,y)] on the base pixel (x, y).
  shape = tf.stack([batch_size, 1, width, depth])
  dy = tf.concat([dy, tf.zeros(shape, img.dtype)], 1)
  dy = tf.reshape(dy, image_shape)

  shape = tf.stack([batch_size, height, 1, depth])
  dx = tf.concat([dx, tf.zeros(shape, img.dtype)], 2)
  dx = tf.reshape(dx, image_shape)

  return dy, dx
'''



'''Simple gradient magnitude difference'''
# Ref.:
# FSIM (ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5705575)
# DBTM (link.springer.com/article/10.1007/s00371-019-01669-8)
# only differentiable gradient magnitude is included, phase congruency is un-differentiable (?)
def grad_mag(img, filter_x, filter_y):

    filter_x = tf.expand_dims(filter_x, axis=2)
    filter_x = tf.expand_dims(filter_x, axis=3)
    filter_y = tf.expand_dims(filter_y, axis=2)
    filter_y = tf.expand_dims(filter_y, axis=3)

    channels = img.shape[3]
    g_all_channel = []
    for i in range(channels):
        img_channel = tf.slice(img, [0, 0, 0, i], [-1, -1, -1, 1])  # 1hw1
        g_x = tf.nn.conv2d(img_channel, filter_x, strides=[1, 1, 1, 1], padding='VALID')
        g_y = tf.nn.conv2d(img_channel, filter_y, strides=[1, 1, 1, 1], padding='VALID')
        g = tf.sqrt(tf.square(g_x) + tf.square(g_y) + 1e-6)  # small epsilon to avoid 0 (?) thus loss NaN
        g_all_channel.append(g)
    g_all_channel = tf.concat(g_all_channel, axis=3)  # 1hw3

    return g_all_channel


def grad_mag_dif(img1, img2, operator='scharr', mode='l1'):

    # filter_x = []
    # filter_y = []
    if operator == 'scharr':
        filter_x = (1 / 16) * tf.constant([[3, 0, -3],
                                          [10, 0, -10],
                                          [3, 0, -3]], dtype=tf.float32)
        filter_y = (1 / 16) * tf.constant([[3, 10, 3],
                                          [0, 0, 0],
                                          [-3, -10, -3]], dtype=tf.float32)
    elif operator == 'prewitt':
        filter_x = (1 / 3) * tf.constant([[1, 0, -1],
                                          [1, 0, -1],
                                          [1, 0, -1]], dtype=tf.float32)
        filter_y = (1 / 3) * tf.constant([[1, 1, 1],
                                          [0, 0, 0],
                                          [-1, -1, -1]], dtype=tf.float32)
    elif operator == 'sobel':
        filter_x = (1 / 4) * tf.constant([[1, 0, -1],
                                          [2, 0, -2],
                                          [1, 0, -1]], dtype=tf.float32)
        filter_y = (1 / 4) * tf.constant([[1, 2, 1],
                                          [0, 0, 0],
                                          [-1, -2, -1]], dtype=tf.float32)
    else:
        sys.exit('Unsupported operator!')

    g1 = grad_mag(img1, filter_x, filter_y)
    g2 = grad_mag(img2, filter_x, filter_y)

    if mode == 'l1':
        return tf.reduce_mean(tf.abs(g1 - g2))
    elif mode == 'l2':
        return tf.reduce_mean(tf.square(g1 - g2))
    else:
        sys.exit('Unsupported mode!')
'''End of simple gradient magnitude difference'''



'''FSITM (TODO, still under development)'''
# phase congruency is un-differentiable (?)
def phasecong100(img, nscale=2, norient=2, min_wave_length=7, mult=2, sigma_onf=0.65):
    # This fun. is transplanted from its MATLAB ver.
    # img:             Input image
    # nscale:          Number of wavelet scales
    # norient:         Number of filter orientations
    # min_wave_length: Wavelength of smallest scale filter
    # mult:            Scaling factor between successive filters
    # sigma_onf:       Ratio of the standard deviation of the Gaussian describing the log Gabor filter's
    #                  transfer function in the frequency domain to the filter center frequency

    h, w = img.shape[1], img.shape[2]
    img = tf.cast(tf.squeeze(img), dtype=tf.complex64)
    '''Fourier transform of image'''
    img_fft = tf.fft2d(img)  # input should be tensor of shape [h,w]?

    zero = tf.zeros([h, w])
    '''Array of convolution results'''
    eo = [nscale, norient]  # list in python =? cell in MATALB?

    '''Matrix for accumulating total energy vector, used for feature orientation and type calculation'''
    energy_v = tf.zeros([h, w, 3])

    '''Set up X and Y matrices with ranges normalised to +/- 0.5'''
    if h / 2 != 0:
        x_range = range(-(h - 1) / 2, (h - 1) / 2, 1 / (h - 1))
    else:
        x_range = range(-h / 2, h / 2, 1 / h)

    if w / 2 != 0:
        y_range = range(-(w - 1) / 2, (w - 1) / 2, 1 / (w - 1))
    else:
        y_range = range(-w / 2, w / 2, 1 / w)

    x, y = tf.stack(x_range), tf.stack(y_range)

    radius = np.sqrt(x ** 2 + y ** 2)  # Matrix values contain *normalised* radius from centre
    theta = np.arctan2(-y, x)  # Matrix values contain polar angle

    '''TO DO'''
    # x, y to tensor

    radius = np.roll(radius, shift=(len(radius) - 1) / 2)  # mimicking 'ifftshift' in MATLAB
    theta = np.roll(theta, shift=(len(theta) - 1) / 2)



def fsitm_gray(hdr, ldr):
    h, w = ldr.shape[1], ldr.shape[2]
    pix_num = h * w
    r = tf.cast(pix_num / 262140, tf.int32)
    if r > 1:
        alpha = 1 - (1 / r)
    else:
        alpha = 0

    hdr_log = tf.log(hdr + 1e-10)
    hdr_log = 255 * norm_to_0_1_tf(hdr_log)

    if alpha != 0:
        phasehdr_ch = phasecong100(hdr, 2, 2, 8, 8)
        phaseldr_ch8 = phasecong100(ldr, 2, 2, 8, 8)
    else:
        Phasehdr_ch = 0
        Phaseldr_ch8 = 0

    phasehdr_log = phasecong100(hdr_log, 2, 2, 2, 2)
    phase_h = alpha * phasehdr_ch + (1 - alpha) * phasehdr_log

    PhaseLDR_CH2 = phasecong100(ldr, 2, 2, 2, 2)
    phase_l = alpha * phaseldr_ch8 + (1 - alpha) * PhaseLDR_CH2

    # index = (phase_l <= 0 & phase_h <= 0) | (phase_l > 0 & phase_h > 0)
    # q = tf.reduce_sum(index) / pix_num




'''(TODO, still under development)'''
def fsitm(hdr, ldr):
    pass
'''End of FSITM'''