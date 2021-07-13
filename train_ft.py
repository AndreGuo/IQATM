# coding: utf-8
from __future__ import print_function
from __future__ import division
from utils.parse_tfrec import *
from loss.cal_loss import *
from utils.lap_pyramid import *
from loss.ssim import *
from loss.color import *
from loss.fsitm import *
from utils.configs import *
import network.net_structure as ns
import time, os, sys
#import matplotlib.pyplot as plt

'''Parameters to modify'''
epochs = 1000 # 50

no_high_branch = False
# no_high_branch = True

no_ft_branch = False
# no_ft_branch = True
'''===================='''


batchnum = round(config.train.total_imgs/config.train.batch_size_ft)
model_ckp = config.model.ckp_path_ft
tfrecord_path = config.model.tfrecord_ft
height = config.data.patch_size_ft_h  # 256
width = config.data.patch_size_ft_w  # 256

tf.logging.set_verbosity(tf.logging.INFO)
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def setconf(layer):
    global model_ckp, tfrecord_path
    if layer == 'high':
        model_ckp = config.model.ckp_path_high
    elif layer == 'bot':
        model_ckp = config.model.ckp_path_bot
    elif layer == 'ft':
        model_ckp = config.model.ckp_path_ft
        tfrecord_path = config.model.tfrecord_ft + '.tfrecord'
    else:
        sys.exit('Wrong requesting layer name!')


def initialzing_input():

    '''reading tfrecord'''
    train_iter = train_iterator_ft(tfrecord_path)
    img, img_high, img_bot, gt = train_iter.get_next()

    return img, img_high, img_bot, gt


def restoreftlayer():
    img, img_h, img_b, gt = initialzing_input()

    '''feeding network'''
    out_h = ns.high_branch(img_h)
    out_b = ns.bot_branch(img_b)
    return out_h, out_b, img, img_h, img_b, gt


def main(goal_epoch, lev):
    setconf('ft')
    with tf.device('/device:GPU:0'):
        with tf.Graph().as_default():
            with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:

                output_high, output_bot, input_, input_high, input_bot, gt = restoreftlayer()
                # arg 'input_' added for unsupervised QA loss

                '''expanding the output of bot branch'''
                h, w = calshape(height, width, lev=3)
                bot_tobe_upsampled = tf.reshape(output_bot, [config.train.batch_size_ft, h, w, 3])

                new_bot = 0
                for index in range(config.train.batch_size_ft):
                    fullsize_bottom = tf.squeeze(tf.slice(bot_tobe_upsampled, [index, 0, 0, 0], [1, -1, -1, -1]))

                    i = tf.constant(0)
                    n = tf.constant(int(lev))
                    fullsize_bottom, i, n = tf.while_loop(cond, body, [fullsize_bottom, i, n],
                                                          shape_invariants=[tf.TensorShape([None, None, 3]),
                                                                            i.get_shape(), n.get_shape()])
                    fullsize_bottom = tf.expand_dims(fullsize_bottom, axis=0)
                    if index == 0:
                        new_bot = fullsize_bottom
                    else:
                        new_bot = tf.concat([new_bot, fullsize_bottom], axis=0) # a batch of then

                if no_high_branch:
                    '''without high branch'''
                    input_ft = input_high + new_bot
                else:
                    '''standard'''
                    input_ft = output_high + new_bot


                loss, output, _= trainlayer(input_ft, gt, sess, input_, output_bot, new_bot, output_high)
                # arg 'input_' added for QA loss, 'output_bot, new_bot, output_high' for observation


                setconf('ft')
                summary = tf.summary.merge_all()
                writer = tf.summary.FileWriter(model_ckp, sess.graph)

                global_step = tf.Variable(0, name="global_step", trainable=False)

                variable_to_train = []
                for variable in tf.trainable_variables():
                    # if not (variable.name.startswith('vgg_16') or
                            # variable.name.startswith('high_level') or variable.name.startswith('bot_level')):
                    if not (variable.name.startswith(config.model.loss_model)):  # avoiding vars in vgg loss
                        variable_to_train.append(variable)
                train_op = tf.train.AdamOptimizer(0.0015).minimize(loss, global_step=global_step,
                                                                                var_list=variable_to_train)
                # default learning rate = 1e-3, epsilon = 1e-8, optimizer = Adam

                variables_to_restore = []
                for v in tf.global_variables():
                    if not (v.name.startswith(config.model.loss_model)):  # avoiding vars in vgg for loss
                        variables_to_restore.append(v)
                saver = tf.train.Saver(variables_to_restore, write_version=tf.train.SaverDef.V2)
                sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

                '''restoring high frequency vars'''
                variables_to_restore = []
                for v in tf.trainable_variables():
                    if v.name.startswith('high'):
                        variables_to_restore.append(v)

                setconf('high')
                saver_h = tf.train.Saver(variables_to_restore, write_version=tf.train.SaverDef.V2)
                ckpt = tf.train.get_checkpoint_state(model_ckp)
                if ckpt and ckpt.model_checkpoint_path:
                    full_path = tf.train.latest_checkpoint(model_ckp)
                    saver_h.restore(sess, full_path)

                '''restoring low frequency vars'''
                variables_to_restore = []
                for v in tf.trainable_variables():
                    if v.name.startswith('bot'):
                        variables_to_restore.append(v)

                setconf('bot')
                saver_l = tf.train.Saver(variables_to_restore, write_version=tf.train.SaverDef.V2)
                ckpt = tf.train.get_checkpoint_state(model_ckp)
                if ckpt and ckpt.model_checkpoint_path:
                    full_path = tf.train.latest_checkpoint(model_ckp)
                    saver_l.restore(sess, full_path)

                '''DEBUGGING to see which loss is NaN'''
                # loss_tmqi = sess.run(loss_tmqi)
                # ls = sess.run(ls)
                # lms = sess.run(lms)
                # lt = sess.run(lt)
                # lh = sess.run(lh)
                # lc = sess.run(lc)
                '''plz modify fuc. 'trainlayer' so it returns specific loss term'''


                setconf('ft')
                # restore variables for training model if the checkpoint file exists.
                epoch = restoreandgetepochs(model_ckp, sess, batchnum, saver)

                ####################
                """Start Training"""
                ####################
                start_time = time.time()
                while True:
                    _, loss_t, step, predict, gtruth = sess.run([train_op, loss, global_step, output, gt])

                    '''exit if loss nan'''  # This doesn't work...
                    '''tf.cond(tf.is_nan(loss_t),
                            lambda: sys.exit('Loss NaN at epoch [%2d], global step: %4d' % (epoch + 1, step)),
                            lambda: None)'''

                    batch_id = int(step % batchnum)
                    elapsed_time = time.time() - start_time
                    start_time = time.time()

                    '''logging'''
                    tf.logging.info("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f, global step: %4d"
                                    % (epoch + 1, batch_id, batchnum, elapsed_time, loss_t, step))

                    '''advance counters'''
                    if batch_id == 0:
                        if epoch >= goal_epoch:
                            break
                        else:
                            '''saving checkpoint'''
                            saver.save(sess, os.path.join(model_ckp, 'model-ft.ckpt'), global_step=step)
                        epoch += 1

                    '''summary'''
                    if step % 20 == 0:
                        tf.logging.info('adding summary...')
                        summary_str = sess.run(summary)
                        writer.add_summary(summary_str, step)
                        writer.flush()

def trainlayer(input_ft, gt, sess, input_, output_bot, new_bot, output_high):  # arg 'input_' added for QA loss
    ##################
    """Feed Network"""
    ##################
    if no_ft_branch:
        '''without ft branch'''
        output = input_ft
    else:
        '''standard'''
        output = ns.ft_branch(input_ft)

    '''calculating L2 regularization value based on trainable weights in the network'''
    l1_reg = 0
    weight_size = 0
    for variable in tf.trainable_variables():
        if not (variable.name.startswith(config.model.loss_model)):
            l1_reg += tf.reduce_sum(tf.abs(variable)) * 2
            weight_size += tf.size(variable)
    l2_reg = l1_reg / tf.to_float(weight_size)

    l1 = tf.reduce_mean(tf.abs(output - gt))
    l2 = tf.reduce_mean((output - gt) ** 2)

    '''perceptual loss'''
    losses = cal_loss(output, gt, config.model.loss_vgg, sess)
    lp = losses.loss_f / 3

    '''supervised SSIM loss (output vs. gt)'''
    # ---tf.image.ssim is only available in higher version of tensorflow---
    # loss_ssim = 1 - tf.reduce_mean(tf.image.ssim(output, gt, max_val=1))  # both linear
    # loss_msssim = 1 - tf.reduce_mean(tf.image.ssim(output, gt, max_val=1))
    # ---if running at tf1.4.0 at py27, use the self-made loss/ssim---
    lssim = 1 - ssim(output, gt, data_range=1)
    lmsssim = 1 - ms_ssim(output, gt, data_range=1)

    '''unsupervised SSIM loss (output vs. input, in a QA like way)'''
    ltmqi = 1 - ms_ssim(output, input_, data_range=1, tmqi=True)

    '''hue shift loss (output vs. input, in a QA like way)'''
    lh = delta_hue_ipt(output, input_, gamut1='709', gamut2='Fairchild Nikon D2x') + 1e-6

    '''color difference loss'''
    lc = delta_e_itp(output, gt, gamut1='709', gamut2='709') + 1e-6

    loss = l1 * 0.6 + lp * 0.3 + lssim * 0.1 + lc * 0.2 + lh * 0.15 + ltmqi * 0.1
    # + l2 * 0.1 + lmsssim * 0.1 + l2_reg * 0.1

    #################
    """Add Summary"""
    #################
    tf.summary.scalar('loss/loss_l2_reg', l2_reg)
    tf.summary.scalar('loss/loss_l1', l1)
    tf.summary.scalar('loss/loss_l2', l2) #0.6
    tf.summary.scalar('loss/loss_p', lp) #0.4
    tf.summary.scalar('loss/loss_ssim', lssim)
    tf.summary.scalar('loss/loss_msssim', lmsssim)
    # tf.summary.scalar('loss/loss_ssim_QA', lsim_QA)
    tf.summary.scalar('loss/loss_tmqi', ltmqi)
    # tf.summary.scalar('loss/loss_b', loss_b)
    tf.summary.scalar('loss/loss_h', lh)
    tf.summary.scalar('loss/loss_c', lc)
    tf.summary.scalar('loss/total_loss', loss)
    tf.summary.image('input', input_, max_outputs=3)
    tf.summary.image('output', output, max_outputs=3)
    tf.summary.image('ground_truth', gt, max_outputs=3)
    # added for observation
    tf.summary.image('input_ft', input_ft, max_outputs=3)
    tf.summary.image('output_bot', output_bot, max_outputs=3)
    tf.summary.image('new_bot', new_bot, max_outputs=3)
    tf.summary.image('output_high', output_high, max_outputs=3)


    return loss, output, gt


def load(ckpt_dir, sess, saver):
    tf.logging.info('reading checkpoint')
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        full_path = tf.train.latest_checkpoint(ckpt_dir)
        global_step = int(full_path.split('/')[-1].split('-')[-1])
        saver.restore(sess, full_path)
        return True, global_step
    else:
        return False, 0


def restoreandgetepochs(ckpt_dir, sess, batchnum, saver):
    status, global_step = load(ckpt_dir, sess, saver)
    if status:
        start_epoch = global_step // batchnum
        tf.logging.info('model restore success')
    else:
        start_epoch = 0
        tf.logging.info("[*] Not find pretrained model!")
    return start_epoch


def calshape(h, w, lev):
    new_h, new_w = h, w
    for i in range(int(lev)):
        new_h = int(new_h / 2)
        new_w = int(new_w / 2)
    return (new_h, new_w)


main(epochs, lev=3)