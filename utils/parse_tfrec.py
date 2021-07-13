import tensorflow as tf
from utils.configs import *


def _parse_func_high(example_proto):
    features = tf.parse_single_example(
        example_proto,
        features={
            'train1': tf.FixedLenFeature(shape=(config.data.patch_size_h*config.data.patch_size_w*3,), dtype=tf.float32),
            'label1': tf.FixedLenFeature(shape=(config.data.patch_size_h*config.data.patch_size_w*3,), dtype=tf.float32),
            'h1': tf.FixedLenFeature([], dtype=tf.int64),
            'w1': tf.FixedLenFeature([], dtype=tf.int64),
        }
    )
    img = features['train1']
    label = features['label1']

    h1 = tf.cast(features['h1'], tf.int32)
    w1 = tf.cast(features['w1'], tf.int32)

    img = tf.reshape(img, [h1, w1, 3])
    label = tf.reshape(label, [h1, w1, 3])

    return img, label


def _parse_func_bot(example_proto):
    features = tf.parse_single_example(
        example_proto,
        features={
            'train2': tf.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing=True),
            'label2': tf.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing=True),
            'h2': tf.FixedLenFeature([], dtype=tf.int64),
            'w2': tf.FixedLenFeature([], dtype=tf.int64),
        }
    )
    img = features['train2']
    label = features['label2']

    h2 = tf.cast(features['h2'], tf.int32)
    w2 = tf.cast(features['w2'], tf.int32)

    img = tf.reshape(img, [h2, w2, 3])
    label = tf.reshape(label, [h2, w2, 3])

    return img, label


def _parse_func_ft(example_proto):
    features = tf.parse_single_example(
        example_proto,
        features={
            'train': tf.FixedLenFeature(shape=(config.data.patch_size_ft_h*config.data.patch_size_ft_w * 3,), dtype=tf.float32),
            'label': tf.FixedLenFeature(shape=(config.data.patch_size_ft_h*config.data.patch_size_ft_w * 3,), dtype=tf.float32),
            'train1': tf.FixedLenFeature(shape=(config.data.patch_size_ft_h*config.data.patch_size_ft_w * 3,), dtype=tf.float32),
            'train2': tf.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing=True),
            'h2': tf.FixedLenFeature([], dtype=tf.int64),
            'w2': tf.FixedLenFeature([], dtype=tf.int64),
        }
    )
    img = features['train']
    label = features['label']
    img_h = features['train1']
    img_b = features['train2']

    h2 = tf.cast(features['h2'], tf.int32)
    w2 = tf.cast(features['w2'], tf.int32)

    img = tf.reshape(img, [config.data.patch_size_ft_h, config.data.patch_size_ft_w, 3])
    label = tf.reshape(label, [config.data.patch_size_ft_h, config.data.patch_size_ft_w, 3])
    img_h = tf.reshape(img_h, [config.data.patch_size_ft_h, config.data.patch_size_ft_w, 3])
    img_b = tf.reshape(img_b, [h2, w2, 3])

    return img, img_h, img_b, label


def _parse_func_test(example_proto):
    feature_labels = {
        'name': tf.FixedLenFeature([], dtype=tf.string),
        'h1': tf.FixedLenFeature([], dtype=tf.int64),
        'w1': tf.FixedLenFeature([], dtype=tf.int64),
        'h2': tf.FixedLenFeature([], dtype=tf.int64),
        'w2': tf.FixedLenFeature([], dtype=tf.int64)
    }

    for l in range(0, 2):
        feature_labels['test{0}'.format(l)] = tf.FixedLenSequenceFeature([], dtype=tf.float32, allow_missing=True)

    features = tf.parse_single_example(
        example_proto,
        features=feature_labels
    )

    name = features['name']
    h1 = tf.cast(features['h1'], tf.int32)
    w1 = tf.cast(features['w1'], tf.int32)
    h2 = tf.cast(features['h2'], tf.int32)
    w2 = tf.cast(features['w2'], tf.int32)

    test0 = features['test0'.format(0)]
    test0 = tf.reshape(test0, [h1, w1, 3])

    test1 = features['test1'.format(1)]
    test1 = tf.reshape(test1, [h2, w2, 3])

    return test0, test1, h1, w1, h2, w2, name


def train_iterator_high(tfrecord):
    data = tf.data.TFRecordDataset(tfrecord)
    data = data.map(_parse_func_high)
    data = data.shuffle(buffer_size=250, reshuffle_each_iteration=True).batch(config.train.batch_size_high).repeat()
    iterater = data.make_one_shot_iterator()
    return iterater


def train_iterator_bot(tfrecord):
    data = tf.data.TFRecordDataset(tfrecord)
    data = data.map(_parse_func_bot)
    data = data.shuffle(buffer_size=250, reshuffle_each_iteration=True).batch(config.train.batch_size_bot).repeat()
    iterater = data.make_one_shot_iterator()
    return iterater


def train_iterator_ft(tfrecord):
    data = tf.data.TFRecordDataset(tfrecord)
    data = data.map(_parse_func_ft)
    data = data.shuffle(buffer_size=500, reshuffle_each_iteration=True).batch(config.train.batch_size_ft).repeat()
    iterater = data.make_one_shot_iterator()
    return iterater


def test_iterator(tfrecord):
    data = tf.data.TFRecordDataset(tfrecord)
    data = data.map(_parse_func_test)
    data = data.repeat()
    iterater = data.make_one_shot_iterator()
    return iterater