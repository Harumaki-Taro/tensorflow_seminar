### mnist.py
### Goal: You can build reusable program of FNN.
### Environment: tensorflow1.7.0, miniconda3-4.3.11, Python3.6.5, OSX10.13.4
### Reference: https://www.tensorflow.org/tutorials/deep_cnn

import os
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('learning_rate', 0.01,
                          """学習率""")
tf.app.flags.DEFINE_float('momentum', 0.9,
                          """慣性係数""")
tf.app.flags.DEFINE_boolean('use_nesterov', True,
                            """Nesterovの加速勾配降下法""")
tf.app.flags.DEFINE_string('mnist_path', '../data/',
                           """mnistデータセットが格納されているパス""")

EXAMPLE_SIZE = 28*28    # 画像のサイズ
LABEL_SIZE = 10         # ラベルの数


def read_data_sets(one_hot=True):
    print('mnist path   : %s' % os.path.abspath(FLAGS.mnist_path))
    return input_data.read_data_sets(FLAGS.mnist_path, one_hot=one_hot)


def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
        tf.summary.histogram(name=name, values=var)

    return var


def _variable_with_weight_decay(name, shape, stddev, wd, seed):
    var = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, seed=seed))

    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('regularizations', weight_decay)

    return var


def affine(x, shape, seed, wd=None, stddev=0.1, bias=0.1):
    if wd == None:
        _wd = 0.0
    else:
        _wd = wd
    print('affine:')
    print('W(shape=%s, stddev=%f, wd=%f, seed=%d)' % (shape, stddev, _wd, seed))
    print('b(shape=%d, init=%f)' % (shape[-1], bias))

    weights = _variable_with_weight_decay('weightes',
                                          shape=shape,
                                          stddev=stddev,
                                          wd=wd,
                                          seed=seed)
    biases = _variable_on_cpu('biases', shape[1], tf.constant_initializer(bias))
    affine = tf.matmul(x, weights) + biases

    return affine


def activate(function, x, name=None):
    if name is not None:
        _name = function.name
    else:
        _name = name
    print('act_func: %s' % _name)

    return function(x)


def loss(logits, labels):
    print('Use "mnist/loss"')

    losses = {}
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')
    losses['cross_entropy'] = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', losses['cross_entropy'])
    tf.summary.scalar('cross_entropy', losses['cross_entropy'])

    if len(tf.get_collection('regularizations')) > 0:
        losses['regularizations'] = tf.add_n(tf.get_collection('regularizations'),
                                             name='total_regularization')
        tf.add_to_collection('losses', losses['regularizations'])
        tf.summary.scalar('regularizations', losses['regularizations'])

    losses['total_loss'] = tf.add_n(tf.get_collection('losses'), name='total_loss')
    tf.summary.scalar('total_loss', losses['total_loss'])

    return losses


def accuracy(logits, labels):
    print('Use "mnist/accuracy"')

    correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    correct_num = tf.reduce_sum(tf.cast(correct, tf.int32))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    return accuracy


def train(losses):
    print('Use "mnist/train"')

    # トレーニングオペレーションの用意（Nesterov Momentum(Sutskever et al., 2013)）
    optimizer = tf.train.MomentumOptimizer(learning_rate=FLAGS.learning_rate,
                                           momentum=FLAGS.momentum,
                                           use_nesterov=FLAGS.use_nesterov)
    train_op = optimizer.minimize(losses['total_loss'])

    print('optimizer    : %s' % optimizer.get_name())
    print('learning_rate: %f' % FLAGS.learning_rate)
    print('momentum     : %f' % FLAGS.momentum)
    print('use_nesterov : ' + str(FLAGS.use_nesterov))

    return train_op
