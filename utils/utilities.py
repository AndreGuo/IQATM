import tensorflow as tf
import scipy.misc
import cv2, os, sys, random
import numpy as np


def norm_to_0_1(img, mode='min-max'):
    img = np.float32(img)
    img_flat = img.flatten()
    max_value, min_value = np.max(img_flat), np.min(img_flat)
    if mode == 'min-max':
        new_img = (img - min_value) / (max_value - min_value)
    elif mode == 'max':
        new_img = img / max_value
    else:
        sys.exit('plz enter right mode name!')
    return new_img

def norm_to_0_255(img, mode='max'):
    img = np.float32(img)
    img_flat = img.flatten()
    max_value, min_value = np.max(img_flat), np.min(img_flat)
    if mode == 'min-max':
        new_img = ((img - min_value) * 255) / (max_value - min_value)
    elif mode == 'max':
        new_img = (img * 255) / max_value
    else:
        sys.exit('plz enter right mode name!')
    return new_img

def norm_to_0_1_tf(img, mode='min-max'):
    max_val, min_val = tf.reduce_max(img), tf.reduce_min(img)
    if mode == 'min-max':
        img_nor = (img - min_val) / (max_val - min_val)
    elif mode == 'max':
        img_nor = img / max_val
    else:
        sys.exit('plz enter right mode name!')
    return img_nor + 1e-6


def norm_0_255_tf(img, mode='max'):
    max_val, min_val = tf.reduce_max(img), tf.reduce_min(img)
    # img = tf.to_float(img)
    if mode == 'min-max':
        img_nor = ((img - min_val) *255) / (max_val - min_val)
    elif mode == 'max':
        img_nor = (img * 255) / max_val
    else:
        sys.exit('plz enter right mode name!')
    return img_nor + 1e-6


'''Cutting extrme value'''
def cut_extreme_value(img, percent, mode='value'):
    if mode == 'value':
        max_val, min_val = np.max(img), np.min(img)
        # smallend = min_val + percent * (max_val - min_val)
        largeend = min_val + (1 - percent) * (max_val - min_val)
    elif mode == 'number':
        [h, w, c] = np.shape(img)
        cut_num = int(h * w * c * percent)
        # smallend = np.partition(img.flatten(), cut_num)[cut_num]
        largeend = np.partition(img.flatten(), -cut_num)[-cut_num]
    else:
        # smallend, largeend = None, None
        sys.exit('plz enter right mode name!')
    # img = np.clip(img, a_min=smallend, a_max=largeend)
    img = np.clip(img, a_min=None, a_max=largeend)
    return img


def save_images_from_event(event_path, tag, output_dir='./'):
    assert(os.path.isdir(output_dir))

    image_str = tf.placeholder(tf.string)
    im_tf = tf.image.decode_image(image_str)

    sess = tf.InteractiveSession()
    with sess.as_default():
        count = 0
        for e in tf.train.summary_iterator(event_path):
            for v in e.summary.value:
                if v.tag.startswith(tag):
                    im = im_tf.eval({image_str: v.image.encoded_image_string})
                    output_fn = os.path.realpath('{}/image_{:05d}.png'.format(output_dir, count))
                    print("Saving '{}'".format(output_fn))
                    sys.stdout.flush()
                    scipy.misc.imsave(output_fn, im)
                    count += 1
    sess.close()


'''This func is under implementation by the original author of DRLTM'''
def demosaicAndSaveImage(pic, y, s):
    pic = np.float32(pic)
    y = np.float32(y)
    lum = 0.2126 * pic[:,:,0] + 0.7152 * pic[:,:,1] + 0.0722 * pic[:,:,2]
    # s = 0.5
    # Demosaicing
    # demosaic_y = pic
    demosaic_y = np.zeros(np.shape(pic))
    demosaic_y[:,:,0] = ((pic[:,:,0]/(lum + 1e-10)) ** s)*y
    demosaic_y[:,:,1] = ((pic[:,:,1]/(lum + 1e-10)) ** s)*y
    demosaic_y[:,:,2] = ((pic[:,:,2]/(lum + 1e-10)) ** s)*y
    return demosaic_y