# coding: utf-8
from __future__ import print_function
from __future__ import division
from utils.parse_tfrec import *
from loss.cal_loss import *
from utils.configs import *
from utils.utilities import *
from loss.ssim import *
from loss.color import *
from loss.fsitm import *
import network.net_structure as ns
import time, os, sys


'''Parameters to modify'''
epochs = 350  # 50
'''===================='''


batchnum = round(config.train.total_imgs/config.train.batch_size_high)
tf.logging.set_verbosity(tf.logging.INFO)
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main(goal_epoch, lev):
    model_ckp = config.model.ckp_path_high
    tfrecord_path = config.model.tfrecord_dual + '.tfrecord'

    with tf.device('/GPU:0'):
        with tf.Graph().as_default():
            with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:

                loss, output, gt = trainlayer(tfrecord_path, sess)

                summary = tf.summary.merge_all()
                writer = tf.summary.FileWriter(model_ckp, sess.graph)

                global_step = tf.Variable(0, name='global_step', trainable=False)

                variable_to_train = []
                for variable in tf.trainable_variables():
                    if not (variable.name.startswith(config.model.loss_model)):
                        variable_to_train.append(variable)
                train_op = tf.train.AdamOptimizer(0.0003).minimize(loss, global_step=global_step,
                                                                 var_list=variable_to_train)
                # default learning rate = 1e-3, optimizer = Adam
                # loss NaN sometimes occur when lr is too small (e.g. 0.0002)

                variables_to_restore = []
                for v in tf.global_variables():
                    if not (v.name.startswith(config.model.loss_model)):
                        variables_to_restore.append(v)
                saver = tf.train.Saver(variables_to_restore, write_version=tf.train.SaverDef.V2)
                sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

                # restore variables for training model if the checkpoint file exists.
                epoch = restoreandgetepochs(model_ckp, sess, batchnum, saver)

                '''DEBUGGING to see which loss is NaN'''
                # l1 = sess.run(l1)
                # l2 = sess.run(l2)
                # lf = sess.run(lf)
                # l2r = sess.run(l2r)
                # ls = sess.run(ls)
                # lsm = sess.run(lms)
                # lt = sess.run(lt)
                # lg = sess.run(lg)
                '''plz modify fuc. 'trainlayer' so it returns specific loss term'''

                ####################
                '''Start Training'''
                ####################
                start_time = time.time()
                while True:
                    _, loss_t, step, predict, gtruth = sess.run([train_op, loss, global_step, output, gt])

                    '''exit if loss nan'''
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
                    if batch_id == batchnum - 1:
                        if epoch >= goal_epoch:
                            break
                        else:
                            '''saving checkpoint'''
                            saver.save(sess, os.path.join(model_ckp, 'model-high.ckpt'), global_step=step)
                        epoch += 1

                    '''summary'''
                    if step % 20 == 0:
                        tf.logging.info('adding summary...')
                        summary_str = sess.run(summary)
                        writer.add_summary(summary_str, step)
                        writer.flush()


def trainlayer(tfrecord_path, sess):

    '''reading tfrecord'''
    train_iter = train_iterator_high(tfrecord_path)
    img_patch, gt_patch = train_iter.get_next()

    '''feeding network'''
    output = ns.high_branch(img_patch)

    '''calculating L2 regularization value based on trainable weights in the network'''
    l1_reg = 0
    weight_size = 0
    for variable in tf.trainable_variables():
        if not (variable.name.startswith(config.model.loss_model)):
            l1_reg += tf.reduce_sum(tf.abs(variable)) * 2
            weight_size += tf.size(variable)
    l2_reg = l1_reg / tf.to_float(weight_size)

    l1 = tf.reduce_mean(tf.abs(output - gt_patch))
    l2 = tf.reduce_mean((output - gt_patch) ** 2)

    '''perceptual loss'''
    losses = cal_loss(output, gt_patch, config.model.loss_vgg, sess)
    lp = losses.loss_f / 3

    '''supervised SSIM loss (output vs. gt)'''
    # ---tf.image.ssim is only available in higher version of tensorflow---
    #  loss_ssim = 1 - tf.reduce_mean(tf.image.ssim(output, gt_patch, max_val=1))  # both linear
    #  loss_msssim = 1 - tf.reduce_mean(tf.image.ssim(output, gt_patch, max_val=1))
    # ---if running at tf1.4.0 at py27, use the self-made loss/ssim---
    lssim = 1 - ssim(norm_to_0_1_tf(output), norm_to_0_1_tf(gt_patch), data_range=1)
    lmsssim = 1 - ms_ssim(norm_to_0_1_tf(output), norm_to_0_1_tf(gt_patch), data_range=1)

    '''unsupervised SSIM(TMQI) loss (output vs. input, in a QA like way)'''
    ltmqi = 1 - ms_ssim(norm_to_0_1_tf(output), norm_to_0_1_tf(img_patch), data_range=1, tmqi=True)

    '''color difference loss'''
    lc = delta_e_itp(output, gt_patch, gamut1='709', gamut2='709')

    loss = l1 * 0.6 + lp * 0.3 + ltmqi * 0.2 + lmsssim * 0.2
    # + lc * 0.3 + lssim * 0.1 + l2 * 0.1 + l2_reg * 0.1

    #################
    """Add Summary"""
    #################
    tf.summary.scalar('loss/loss_l1', l1)#0.5
    tf.summary.scalar('loss/loss_l2', l2)
    tf.summary.scalar('loss/loss_l2_reg', l2_reg)
    tf.summary.scalar('loss/loss_f', lp)#0.5
    tf.summary.scalar('loss/loss_ssim', lssim)
    tf.summary.scalar('loss/loss_msssim', lmsssim)
    tf.summary.scalar('loss/loss_c', lc)
    # tf.summary.scalar('loss/loss_tmqi', ltmqi)
    tf.summary.scalar('loss/total_loss', loss)
    tf.summary.image('input', img_patch, max_outputs=3)
    tf.summary.image('output', output, max_outputs=3)
    tf.summary.image('ground_truth', gt_patch, max_outputs=3)

    return loss, output, gt_patch


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


def restoreandgetepochs(ckpt_dir, sess, batchnum, savaer):
    status, global_step = load(ckpt_dir, sess, savaer)
    if status:
        start_epoch = global_step // batchnum
        tf.logging.info('model restore success')
    else:
        start_epoch = 0
        tf.logging.info("[*] Not find pre-trained model!")
    return start_epoch


main(epochs, lev=3)