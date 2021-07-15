from __future__ import print_function, division
import os, sys, cv2, glob
from utils.configs import *
from utils.utilities import *
from utils.lap_pyramid import *
from utils.parse_tfrec import *
import network.net_structure as ns
import imageio as io


'''Params. for ablation studies'''
level = config.dev.level_num
data_mode = config.dev.data_domain

# no_hight_branch = True
no_hight_branch = False

# no_ft_branch = True
no_ft_branch = False
'''===================='''


tf.logging.set_verbosity(tf.logging.INFO)
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


'''padding the bottom layer to refrain the ripple-boarder effect'''
pad_width = 6


def initialzing_input():

    '''reading test tfrecord'''
    test_iter = test_iterator(tfrecord_path)
    high, bot, h1, w1, h2, w2, name = test_iter.get_next()

    # hwc -> 1hwc
    high = tf.expand_dims(high, axis=0)
    bot = tf.expand_dims(bot, axis=0)

    '''padding the bottom layer to refrain the ripple-boarder effect (only for bot)'''
    bot_padded = tf.pad(bot, [[0, 0], [pad_width, pad_width], [pad_width, pad_width], [0, 0]], "REFLECT")

    return high, bot_padded, h1, w1, h2, w2, name


def restoreftlayer():
    high, bot, h1, w1, h2, w2, name = initialzing_input()

    ##################
    """Feed Network"""
    ##################
    out_h = ns.high_branch(high)
    out_b = ns.bot_branch(bot)

    return out_h, out_b, high, bot, h1, w1, h2, w2, name


def main(lev):
    global model_ckp, tfrecord_path, hdr_dir

    hdr_dir = config.test.hdr_path
    tfrecord_path = config.test.tfrecord_test + '.tfrecord'
    model_ckp = config.model.ckp_path_ft

    with tf.device('/cpu:0'):
        with tf.Graph().as_default():
            with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
                output_high, output_bot, input_high, input_bot, h1, w1, h2, w2, name = restoreftlayer()

                '''recovering original bot shape'''
                # output_bot = tf.slice(output_bot, [0, pad_width, pad_width, 0], [-1, h2, w2, -1])
                out_b_shape = tf.shape(output_bot)
                output_bot = tf.slice(output_bot, [0, pad_width, pad_width, 0],
                                      [-1, out_b_shape[1] - pad_width * 2, out_b_shape[2] - pad_width * 2, -1])

                bot_h, bot_w = calshape_tf(h1, w1, lev)
                bot_tobe_upsampled = tf.reshape(tf.squeeze(output_bot), [bot_h, bot_w, 3])

                i = tf.constant(0)
                n = tf.constant(int(lev))
                fullsize_bottom, i, n = tf.while_loop(cond, body, [bot_tobe_upsampled, i, n],
                                                      shape_invariants=[tf.TensorShape([None, None, 3]), i.get_shape(),
                                                                        n.get_shape()])

                # fullsize_bottom = tf.slice(fullsize_bottom, [0, 0, 0], [h1, w1, 3])
                fullsize_bottom = tf.expand_dims(fullsize_bottom, axis=0)


                if no_hight_branch:
                    '''without high branch'''
                    ft_input = input_high + fullsize_bottom
                else:
                    '''standard'''
                    ft_input = output_high + fullsize_bottom

                if no_ft_branch:
                    '''without ft branch'''
                    output = ft_input
                else:
                    '''standard'''
                    output = ns.ft_branch(ft_input)


                '''loading network'''
                variables_to_restore = []
                for v in tf.global_variables():
                    if not (v.name.startswith(config.model.loss_model)):
                        variables_to_restore.append(v)

                saver = tf.train.Saver(variables_to_restore, write_version=tf.train.SaverDef.V2)
                ckpt = tf.train.get_checkpoint_state(model_ckp)
                if ckpt and ckpt.model_checkpoint_path:
                    full_path = tf.train.latest_checkpoint(model_ckp)
                    saver.restore(sess, full_path)

                counter = 0
                num_test_imgs = len(glob.glob(hdr_dir + '*.{}'.format(config.data.appendix_hdr)))

                while counter < num_test_imgs:
                    predict, filename = sess.run([output, name])
                    # predict, filename = sess.run([input_bot, name])
                    predict = np.squeeze(predict)
                    # predict = cut_extreme_value(predict, 0.001, mode='number')

                    predict = norm_to_0_1(predict)

                    if data_mode == 'gamma':
                        pass
                    if data_mode == 'linear':
                        '''transferring linear output to non-linear'''
                        predict = predict ** 0.4545  # 0.4545 for gamma2.2(sRGB) & 0.4167 for gamma2.4(BT.1886)
                    predict = norm_to_0_255(predict)  # min-max normalization

                    io.imwrite(config.test.result + filename + '.jpg', predict)

                    counter += 1


def calshape_tf(h, w, lev):
    new_h, new_w = h, w
    for i in range(int(lev)):
        new_h = tf.cast(tf.ceil(tf.divide(new_h, 2)), tf.int32)
        new_w = tf.cast(tf.ceil(tf.divide(new_w, 2)), tf.int32)
        # new_h = tf.cast(tf.divide(new_h, 2), tf.int32)
        # new_w = tf.cast(tf.divide(new_w, 2), tf.int32)
    return new_h, new_w


main(level)