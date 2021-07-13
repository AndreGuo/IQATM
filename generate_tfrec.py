from __future__ import print_function
import glob, os, sys, cv2, random, math
import tensorflow as tf
import numpy as np
import imageio as io
from utils.lap_pyramid import *
from utils.configs import *
from utils.utilities import *
import copy  # only for py27, no need in py37


'''MAKE SURE phase == 'testing' when you only want to test our method'''
# phase = 'training'
phase = 'testing'

'''Set to True when your GPU is out of memory'''
# test_resize_to_half = True
test_resize_to_half = False
'''===================='''

'''Param. useful only when phase == 'training' '''
# ft = False  # for training high & bottom layer (Step. 1)
ft = True  # for training all layer (fine tuning, Step. 2)
'''===================='''


'''3 params. below are only for developing'''
# mode = 'linear'
mode = 'gamma'

# filter_mode = 'gaussian'
filter_mode = 'bilateral'

# fuse_mid_layers = True
fuse_mid_layers = False
'''===================='''


apdx_sdr = config.data.appendix_sdr
apdx_hdr = config.data.appendix_hdr


def gen_train_tfrec(fine_tuning=False):
    if fine_tuning:
        tfrecord_path = config.model.tfrecord_ft + '.tfrecord'
    else:
        tfrecord_path = config.model.tfrecord_dual + '.tfrecord'

    with tf.python_io.TFRecordWriter(tfrecord_path) as tfrecord_writer:

        '''getting HDR images list'''
        file_list = glob.glob(config.data.hdr_path + '*.{}'.format(apdx_hdr))

        '''finding corresponding SDR images'''
        pair_num = len(file_list)

        for index in range(len(file_list)):
            cur_path = file_list[index]
            print('Processing Image -> ' + cur_path, ' %d / %d' % (index + 1, pair_num))
            file_name = os.path.splitext(os.path.basename(cur_path))
            sdr_path = config.data.sdr_path + file_name[0] + '.' + apdx_sdr

            '''reading HDR & SDR images using imageio'''
            hdr_img = io.imread(cur_path)  # linear [0,1]
            sdr_img = io.imread(sdr_path)  # sRGB [0,255]

            '''checking shape'''
            assert(np.shape(hdr_img) == np.shape(sdr_img))

            '''cutting HDR extreme value'''
            hdr_img = cut_extreme_value(hdr_img, percent=0.003, mode='number')

            '''normalizing linear HDR images'''
            hdr_ln = norm_to_0_1(hdr_img)  # it's min/max normalized

            if mode == 'linear':
                '''linearizing sRGB SDR images to [0,1]'''
                # normalizing via 'sdr_img/255' will cause data error
                sdr_ln = norm_to_0_1(sdr_img) ** 2.2  # 2.4 for BT.1886 image
            elif mode == 'gamma':
                '''non-linearizing linear HDR to sRGB'''
                hdr_ln = hdr_ln ** 0.4545 # 0.4167 for BT.1886 image
                sdr_ln = norm_to_0_1(sdr_img)
            else:
                sys.exit('unsupported mode name')

            '''randomly cropping images'''
            if fine_tuning:
                patch_h = config.data.patch_size_ft_h
                patch_w = config.data.patch_size_ft_w
            else:
                patch_h = config.data.patch_size_h
                patch_w = config.data.patch_size_w
            hdr_patches, sdr_patches = randomly_cropping(hdr_ln, sdr_ln,
                                                         config.data.patch_ratio_x,config.data.patch_ratio_y,
                                                         patch_h, patch_w, config.data.patch_per_img - 1)

            '''adding last patch from resizing'''
            hdr_patches.append(cv2.resize(hdr_ln, (patch_h, patch_w)))
            sdr_patches.append(cv2.resize(sdr_ln, (patch_h, patch_w)))

            '''checking length'''
            assert(len(hdr_patches) == len(sdr_patches))

            #  hdr_pyr_patches, sdr_pyr_patches = hdr_patches.copy(), sdr_patches.copy()  # 'copy' for py37
            hdr_pyr_patches, sdr_pyr_patches = copy.copy(hdr_patches), copy.copy(sdr_patches)  # 'copy' for py27

            '''creating 'laplacian' pyramid for input HDR images'''
            for i in range(len(hdr_pyr_patches)):
                hdr_pyr_patches[i] = lpyr_gen(hdr_pyr_patches[i], mode=filter_mode)
                # making it only high & bot layers
                if not fuse_mid_layers:
                    hdr_pyr_patches[i] = extract_to_dual_layers(hdr_pyr_patches[i])
                hdr_pyr_patches[i] = fuse_to_dual_layers(hdr_pyr_patches[i], mode=filter_mode)

            '''creating 'laplacian' pyramid for target SDR images (when training high & bot branch)'''
            if not fine_tuning:
                for i in range(len(hdr_pyr_patches)):
                    sdr_pyr_patches[i] = lpyr_gen(sdr_pyr_patches[i], mode=filter_mode)
                    if not fuse_mid_layers:
                        sdr_pyr_patches[i] = extract_to_dual_layers(sdr_pyr_patches[i])
                    sdr_pyr_patches[i] = fuse_to_dual_layers(sdr_pyr_patches[i], mode=filter_mode)

                    '''checking shape'''
                    assert (np.shape(hdr_pyr_patches[i]) == np.shape(sdr_pyr_patches[i]))

            '''checking length'''
            assert (len(hdr_pyr_patches) == len(sdr_pyr_patches))

            '''writing in tfrecord'''
            patch_length = len(hdr_patches)
            for i in range(0, patch_length):
                print('\r-- processing images patches %d / %d' % (i + 1, patch_length), end='')
                sys.stdout.flush()
                example = pack_example_train(hdr_pyr_patches[i], sdr_pyr_patches[i], hdr_patches[i], sdr_patches[i], fine_tuning=ft)
                tfrecord_writer.write(example.SerializeToString())
            print('Images Processed -> ' + cur_path, ' %d / %d' % (index + 1, pair_num))
            print('\n')


def gen_test_tfrec():
    tfrecord_path = config.test.tfrecord_test + '.tfrecord'
    with tf.python_io.TFRecordWriter(tfrecord_path) as tfrecord_writer:

        '''getting HDR images list'''
        file_list = glob.glob(config.test.hdr_path + '*.{}'.format(apdx_hdr))
        length_images = len(file_list)
        for index in range(len(file_list)):
            cur_path = file_list[index]
            print( 'Processing Image -> ' + cur_path, ' %d / %d' % (index + 1, length_images))
            file_name = os.path.splitext(os.path.basename(cur_path))[0]

            '''reading in the HDR images'''
            hdr_img = io.imread(cur_path)

            height, width = hdr_img.shape[:2]
            '''RESIZE NEEDED if OMM when testing'''
            if test_resize_to_half:
                size = (int(width * 0.5), int(height * 0.5))
                hdr_img = cv2.resize(hdr_img, size)  # cv2.resize(hdr_img, (config.data.width / 2, config.data.height / 2))

            '''CROPPING NEEDED if h or w can't be divided by 8'''
            if width % 8 != 0 or height % 8 != 0:
                width_8, height_8 = int((width // 8) * 8), int((height // 8) * 8)
                hdr_img = hdr_img.copy()[0:height_8, 0:width_8, :]

            '''cutting extreme value'''
            hdr_img = cut_extreme_value(hdr_img, percent=0.003, mode='number')

            '''normalizing linear HDR images'''
            hdr_ln = norm_to_0_1(hdr_img)  # it's min/max normalized

            if mode == 'linear':
                pass
            elif mode == 'gamma':
                '''non-linearizing linear HDR to sRGB'''
                hdr_ln = hdr_ln ** 0.4545  # 0.4167 for BT.1886 image
            else:
                sys.exit('unsupported mode name')

            '''creating 'laplacian' pyramid for input HDR images'''
            hdr_pyr = lpyr_gen(hdr_ln, mode=filter_mode)
            if not fuse_mid_layers:
                hdr_pyr = extract_to_dual_layers(hdr_pyr)
            hdr_pyr = fuse_to_dual_layers(hdr_pyr, mode=filter_mode)

            '''writing in tfrecord'''
            example = pack_example_test(hdr_pyr, file_name)
            tfrecord_writer.write(example.SerializeToString())
            print('Image Processed-> ' + cur_path, ' %d / %d' % (index+1, length_images))
            print('\n')


def pack_example_train(img_pyr_patch, label_pyr_patch, img_patch, label_patch, fine_tuning):
    features = {}

    if fine_tuning:
        label_patch = np.reshape(label_patch, -1)
        features['label'] = tf.train.Feature(float_list=tf.train.FloatList(value=label_patch))

        '''adding 'img' for calculating unsupervised IQA loss (out vs. img rather than out vs. label)'''
        img_patch = np.reshape(img_patch, -1)
        features['train'] = tf.train.Feature(float_list=tf.train.FloatList(value=img_patch))

    else:
        label_high_patch = np.reshape(label_pyr_patch[0], -1)
        features['label1'] = tf.train.Feature(float_list=tf.train.FloatList(value=label_high_patch))
        label_bot_patch = np.reshape(label_pyr_patch[1], -1)
        features['label2'] = tf.train.Feature(float_list=tf.train.FloatList(value=label_bot_patch))

    h1, w1, _ = img_pyr_patch[0].shape  # high shape (of patch)
    h2, w2, _ = img_pyr_patch[1].shape  # bot shape (of patch)

    features['h1'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[h1]))
    features['w1'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[w1]))
    features['h2'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[h2]))
    features['w2'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[w2]))

    img_high_patch = np.reshape(img_pyr_patch[0], -1)
    features['train1'] = tf.train.Feature(float_list=tf.train.FloatList(value=img_high_patch))

    img_bot_patch = np.reshape(img_pyr_patch[1], -1)
    features['train2'] = tf.train.Feature(float_list=tf.train.FloatList(value=img_bot_patch))

    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example


def pack_example_test(img, name):

    h1, w1, _ = img[0].shape  # high shape (of full-size input)
    h2, w2, _ = img[1].shape  # bot shape (of full-size input)

    '''storing metadata'''
    features = {}

    features['name'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[name.encode('ascii')]))
    # encoding str to bytes WHEN PYTHON3, see: 'cnblogs.com/blili/p/11798504.html'
    features['h1'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[h1]))
    features['w1'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[w1]))
    features['h2'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[h2]))
    features['w2'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[w2]))

    for l in range(0, len(img)):
        img[l] = np.reshape(img[l], -1)
        features['test{0}'.format(l)] = tf.train.Feature(float_list=tf.train.FloatList(value=img[l]))

    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example


def randomly_cropping(img, label, x, y, size_h, size_w, patch_per_img):
    hdrpatches = []
    sdrpatches = []
    h, w, _ = np.shape(img)

    '''when img is too small, resizing & duplicating instead of cropping'''
    # for dataset like Waterloo IVC MEFI (usually around 710*310)
    if h < size_h or w < size_w:
        for i in range(patch_per_img):
            img_resize = cv2.resize(img, (size_h, size_w))
            hdrpatches.append(img_resize)
            label_resize = cv2.resize(label, (size_h, size_w))
            sdrpatches.append(label_resize)
    else:
        for i in range(patch_per_img):
            '''randomly getting the position of cropping starting point (from up & left)'''
            rand_coe_h = random.random() * (y - x) + x
            rand_coe_w = random.random() * (y - x) + x
            rand_h = int(h * rand_coe_h)
            rand_w = int(w * rand_coe_w)

            '''randomly generated coordinates are limited in:'''
            coor_h = h - rand_h  # h -> [0, coor_h]
            coor_w = w - rand_w  # w -> [0, coor_w]

            '''getting x and y starting point of the patch'''
            coor_x = int(random.random() * coor_h)
            coor_y = int(random.random() * coor_w)

            '''create HDR patches'''
            img_patch = img[coor_x:(coor_x + rand_h), coor_y:(coor_y + rand_w), :]
            img_resize = cv2.resize(img_patch, (size_h, size_w))
            hdrpatches.append(img_resize)

            '''creating SDR patches'''
            label_patch = label[coor_x:(coor_x + rand_h), coor_y:(coor_y + rand_w), :]
            label_resize = cv2.resize(label_patch, (size_h, size_w))
            sdrpatches.append(label_resize)
    return hdrpatches, sdrpatches


if phase == 'training':
    if isinstance(ft, bool):
        gen_train_tfrec(fine_tuning=ft)
    else:
        sys.exit('Please enter the right ft name!')
elif phase == 'testing':
    gen_test_tfrec()
else:
    sys.exit('Please enter the right phase name!')