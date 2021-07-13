import tensorflow as tf

def high_branch(input):
    c = 24
    with tf.variable_scope(name_or_scope="high_level"):
        res = norm(conv_lrelu(input=input, in_c=3, out_c=c, padding='SAME', name='conv1'), mode='IN', name='c1_n')
        output = norm(conv_lrelu(input=res, in_c=c, out_c=c, padding='SAME', name='conv2'), mode='IN', name='c2_n')
        output = norm(conv_lrelu(input=output, in_c=c, out_c=c, padding='SAME', name='conv3'), mode='IN', name='c3_n')
        output = norm(conv_lrelu(input=output, in_c=c, out_c=c, padding='SAME', name='conv4'), mode='IN', name='c4_n')
        output += res
        output = conv_sigmoid(input=output, in_c=c, out_c=3, filter_size=1, padding='SAME', name='c5_sig')
    return output + input

def bot_branch(input):
    c = 24
    with tf.variable_scope(name_or_scope="bot_level"):
        output = norm(conv_lrelu(input=input, in_c=3, out_c=c, padding='SAME', name='conv1'), mode='BN', name='c1_n')
        output = norm(conv_lrelu(input=output, in_c=c, out_c=c, padding='SAME', name='conv2'), mode='BN', name='c2_n')
        output = trigroup_resblock(output, in_c=c, out_c=c, name='3pass_res')
        output = norm(conv_lrelu(input=output, in_c=c, out_c=c, padding='SAME', name='conv2'), mode='BN', name='c3_n')
        output = conv_sigmoid(input=output, in_c=c, out_c=3, filter_size=1, padding='SAME', name='c4_sig')
    return output + input


def ft_branch(input):
    c = 24
    with tf.variable_scope(name_or_scope="ft_merger"):
        output = norm(conv_lrelu(input, in_c=3, out_c=c, padding='SAME', name='conv1'), mode='IN', name='c1_n')
        output = norm(conv_lrelu(output, in_c=c, out_c=c, padding='SAME', name='conv2'), mode='IN', name='c2_n')
        output = trigroup_resblock(output, in_c=c, out_c=c, name='3pass_res1')
        output = trigroup_resblock(output, in_c=c, out_c=c, name='3pass_res2')
        output = norm(conv_lrelu(output, in_c=c, out_c=c, padding='SAME', name='conv3'), mode='IN', name='c3_n')
        output = conv_sigmoid(output, in_c=c, out_c=3, filter_size=1, padding='SAME', name='c5_sig')
        return output


'''multi(3)-path (with different k_size) res_block'''
def tripass_resblock(input, in_c, out_c, name='3pass_res'):
    p1 = norm(conv_lrelu(input, in_c, in_c, name=name + 'p1', filter_size=1, strides=1, padding='SAME'), mode='IN', name='p1_n')
    p2 = norm(conv_lrelu(input, in_c, in_c, name=name + 'p2', filter_size=3, strides=1, padding='SAME'), mode='IN', name='p2_n')
    p3 = norm(conv_lrelu(input, in_c, in_c, name=name + 'p3', filter_size=5, strides=1, padding='SAME'), mode='IN', name='p3_n')
    concat = norm(conv_lrelu(tf.concat([p1, p2, p3], 3), in_c * 3, out_c, name=name + 'cc_n', filter_size=3, strides=1, padding='SAME'), mode='IN')
    return concat + input


'''multi(3)-path (with different k_size) res_block'''
def trigroup_resblock(input, in_c, out_c, name='3group_res'):
    g_c = in_c / 3
    g1 = tf.slice(input, [0, 0, 0, 0], [-1, -1, -1, g_c])
    g2 = tf.slice(input, [0, 0, 0, g_c], [-1, -1, -1, g_c])
    g3 = tf.slice(input, [0, 0, 0, g_c * 2], [-1, -1, -1, g_c])
    g1 = norm(conv_lrelu(g1, g_c, g_c, name=name + 'g1', filter_size=1, strides=1, padding='SAME'), mode='IN', name='g1_n')
    g2 = norm(conv_lrelu(g2, g_c, g_c, name=name + 'g2', filter_size=3, strides=1, padding='SAME'), mode='IN', name='g2_n')
    g3 = norm(conv_lrelu(g3, g_c, g_c, name=name + 'g3', filter_size=5, strides=1, padding='SAME'), mode='IN', name='g3_n')
    concat = norm(conv_lrelu(tf.concat([g1, g2, g3], 3), in_c, out_c, name=name + 'cc_n', filter_size=3, strides=1, padding='SAME'), mode='IN')
    return concat + input


'''normalization'''
# Ref.:
# 'cnblogs.com/hellcat/p/7380022.html'
# 'cnblogs.com/hellcat/p/9735041.html'
def norm(input, mode='BN', name='batch_norm'):
    # input_shape = input.get_shape()

    axes = []
    if mode == 'BN':
        axes = [0, 1, 2]  # for NHWC
        name = 'batch_norm'
        # axes = list(range(len(input_shape.as_list()) - 1))
    elif mode == 'LN':
        axes = [1, 2, 3]  # for NHWC
        name = 'layer_norm'
        # axes = list(range(1, len(input_shape.as_list())))
    elif mode == 'IN':
        name = 'instance_norm'
        axes = [1, 2]  # for NHWC
        # axes = list(range(1, len(input_shape.as_list()) - 1))

    mean, variance = tf.nn.moments(input, axes=axes, keep_dims=True)
    output = tf.nn.batch_normalization(input, mean, variance, offset=None, scale=None, variance_epsilon=1e-6, name=name)
    return output


'''initializing filter weight'''
def weight_init(shape, name=None, stddev=0.1):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial, name=name)


'''initializing filter output bias'''
def bias_init(shape, name=None):
    initial = tf.constant(0.005, shape=shape)
    return tf.Variable(initial, name=name)


'''initialized conv2d layer with leaky ReLU activation'''
def conv_lrelu(input, in_c, out_c, name, filter_size=3, strides=1, padding='SAME'):
    w = weight_init([filter_size, filter_size, in_c, out_c], name="{0}_weight".format(name))
    b = bias_init([out_c], name="{0}_bias".format(name))
    output = tf.nn.conv2d(input=input,
                           filter=w,
                           padding=padding,
                           strides=[1, strides, strides, 1],
                           name="{0}_conv".format(name),)
    output = tf.nn.leaky_relu(output + b, alpha=0.3, name="{}_leaky_relu".format(name))
    return output


'''initialized conv2d layer with sigmoid activation'''
def conv_sigmoid(input, in_c, out_c, name, filter_size=3, strides=1, padding='SAME'):
    w = weight_init([filter_size, filter_size, in_c, out_c], name="{0}_weight".format(name))
    b = bias_init([out_c], name="{0}_bias".format(name))
    output = tf.nn.conv2d(input=input,
                           filter=w,
                           padding=padding,
                           strides=[1, strides, strides, 1],
                           name="{0}_conv".format(name),)
    output = tf.nn.sigmoid(output + b, name='{}_sigmoid'.format(name))
    return output