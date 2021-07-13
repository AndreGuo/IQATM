import cv2
import tensorflow as tf
import numpy as np


def bilateral_pyr(img, lev):
    img = img.astype(np.float32)
    b_pyr = [img]
    cur_b = img
    bfd_b_pyr_1 = cv2.bilateralFilter(cur_b, 5, 75, 75)
    h, w = np.shape(cur_b)[0], np.shape(cur_b)[1]
    # down-scaling the first layer using bilateral filter
    h, w = h // 2, w // 2
    cur_b = cv2.bilateralFilter(cur_b, 5, 75, 75)
    cur_b = cv2.resize(cur_b, dsize=(w, h), interpolation=cv2.INTER_AREA)
    b_pyr.append(cur_b)
    # down-scaling the rest layers using gaussian filter
    for index in range(lev-1):
        cur_b = cv2.pyrDown(cur_b)
        b_pyr.append(cur_b)
    # feeding back bilateral filtered first layer
    return b_pyr, bfd_b_pyr_1


def gaussian_pyr(img, lev):
    img = img.astype(np.float32)
    g_pyr = [img]
    cur_g = img
    for index in range(lev):
        cur_g = cv2.pyrDown(cur_g)
        g_pyr.append(cur_g)
    return g_pyr


'''generating the "laplacian" pyramid from an image with specified number of levels'''
# to create bot layer, h & w of img (patch) is required to be multiples of 2**lev
def lpyr_gen(img, lev=3, mode='gaussian'):
    # @ param mode: 'gaussian'as DRLTM; 'bilateral' as IQATM
    img = img.astype(np.float32)  # this changes whatever the img dtype to 'float64'

    if mode == 'gaussian':
        g_pyr = gaussian_pyr(img, lev)
        l_pyr = []
        h, w = np.shape(g_pyr[0])[0], np.shape(g_pyr[0])[1]
        for index in range(lev):
            cur_g = g_pyr[index]
            next_g = cv2.pyrUp(g_pyr[index + 1], dstsize=(w, h))
            cur_l = cv2.subtract(cur_g, next_g)
            l_pyr.append(cur_l)
            h, w = h // 2, w // 2
        l_pyr.append(g_pyr[-1])
        return l_pyr

    elif mode == 'bilateral':
        b_pyr, bfd_b_pyr_1 = bilateral_pyr(img, lev)
        l_pyr = []
        l_pyr_1 = cv2.subtract(b_pyr[0], bfd_b_pyr_1)
        l_pyr.append(l_pyr_1)
        h, w = np.shape(b_pyr[1])[0], np.shape(b_pyr[1])[1]
        for index in range(lev-1):
            cur_b = b_pyr[index + 1]
            #next_b = cv2.resize(b_pyr[index + 1], dsize=(w, h), interpolation=cv2.INTER_AREA)
            #next_b = cv2.bilateralFilter(next_b, 5, 75, 75)
            next_b = cv2.pyrUp(b_pyr[index + 2], dstsize=(w, h))
            # cur_l = cv2.subtract(cur_b, next_b)
            cur_l = cur_b - next_b
            l_pyr.append(cur_l)
            h, w = h // 2, w // 2
        l_pyr.append(b_pyr[-1])
        return l_pyr


def lpyr_upsample(l_img, scope, mode='gaussian'):
    cur_l = l_img
    # repeat x2 up-sample n times, n is on lev
    # scope = dh // np.shape(cur_l)[0]
    # for i in range(scope-1):
    for i in range(scope):
        if mode == 'gaussian':
            cur_l = cv2.pyrUp(cur_l)
        elif mode == 'bilateral':
            cur_l = cv2.resize(cur_l, dsize=(np.shape(cur_l)[1]*2, np.shape(cur_l)[0]*2), interpolation=cv2.INTER_AREA)
            cur_l = cv2.bilateralFilter(cur_l, 5, 75, 75)
        # h, w = h // 2, w // 2
    return cur_l


'''making it only 2 layers: top & buttom, while adding up all layer except buttom'''
def fuse_to_dual_layers(l_pyr, mode='gaussian'):
    # making all levels of pyramid (excluding the bottom layer) to the same size as the largest one
    lev = len(l_pyr)
    levels = []
    cur_l = []
    h, w = np.shape(l_pyr[0])[0], np.shape(l_pyr[0])[1]  # dst size
    for index in range(lev-1):
        levels.append(l_pyr[index].shape)
        aligned = lpyr_upsample(l_pyr[index], scope=index, mode=mode)
        cur_l.append(aligned)
    cur_l.append(l_pyr[lev-1])
    # adding all high frequency layer up (low layer unchanged), making it only 2 layers
    py_layers = cur_l
    freq_layer = 0
    bottom_layer = py_layers[-1]
    freq_layers = py_layers[:-1]  # all but last one
    for item in range(0, len(freq_layers)):
        freq_layer += freq_layers[item]
    dual_layers = [freq_layer, bottom_layer]
    return dual_layers

'''making it only 2 layers: top & buttom, without adding all high frequency layer up'''
def extract_to_dual_layers(l_pyr):
    dual_layers = [l_pyr[0], l_pyr[-1]]
    return dual_layers


# Below are 3 functions used in tf.loop in test.py
'''func. as upsampling'''
def dilatezeros(imgs): # for [h,w] (orginal), not [h,w,3]
    zeros = tf.zeros_like(imgs)
    column_zeros = tf.reshape(tf.stack([imgs, zeros], 2), [-1, tf.shape(imgs)[1] + tf.shape(zeros)[1]])[:, :-1]

    row_zeros = tf.transpose(column_zeros)

    zeros = tf.zeros_like(row_zeros)
    dilated = tf.reshape(tf.stack([row_zeros, zeros], 2), [-1, tf.shape(row_zeros)[1] + tf.shape(zeros)[1]])[:, :-1]
    dilated = tf.transpose(dilated)

    paddings = tf.constant([[0, 1], [0, 1]])
    dilated = tf.pad(dilated, paddings, "REFLECT")
    return dilated


def applygaussian(imgs): # for [1,h,w,1(not 3)]
    gauss_f = tf.constant([[1./256., 4./256., 6./256., 4./256., 1./256.],
                        [4./256., 16./256., 24./256., 16./256., 4./256.],
                        [6./256., 24./256., 36./256., 24./256., 6./256.],
                        [4./256., 16./256., 24./256., 16./256., 4./256.],
                        [1./256., 4./256., 6./256., 4./256., 1./256.]])
    gauss_f = tf.expand_dims(gauss_f, axis=2)
    gauss_f = tf.expand_dims(gauss_f, axis=3)

    result = tf.nn.conv2d(imgs, gauss_f * 4, strides=[1, 1, 1, 1], padding="VALID")
    result = tf.squeeze(result, axis=0)
    return result


def body(output_bot, i, n):  # output_bot: [h,w,c(3)]
    img_r = tf.squeeze(tf.slice(output_bot, [0, 0, 0], [-1, -1, 1]))
    img_g = tf.squeeze(tf.slice(output_bot, [0, 0, 1], [-1, -1, 1]))
    img_b = tf.squeeze(tf.slice(output_bot, [0, 0, 2], [-1, -1, 1]))
    # upsampling
    img_r = tf.expand_dims(dilatezeros(img_r), 2)  # [h,c,1]
    img_g = tf.expand_dims(dilatezeros(img_g), 2)
    img_b = tf.expand_dims(dilatezeros(img_b), 2)
    # end of upsampling
    img_r = tf.expand_dims(img_r, 0)  # [1,h,c,1]
    img_g = tf.expand_dims(img_g, 0)
    img_b = tf.expand_dims(img_b, 0)
    img_r = applygaussian(tf.pad(img_r, [[0, 0], [2, 2], [2, 2], [0, 0]], "REFLECT"))  # [h,c,1]
    img_g = applygaussian(tf.pad(img_g, [[0, 0], [2, 2], [2, 2], [0, 0]], "REFLECT"))
    img_b = applygaussian(tf.pad(img_b, [[0, 0], [2, 2], [2, 2], [0, 0]], "REFLECT"))
    output_bot = tf.concat([img_r, img_g, img_b], 2)  # [h,c,3]
    return output_bot, tf.add(i, 1), n


def cond(output_bot, i, n):
    return tf.less(i, n)