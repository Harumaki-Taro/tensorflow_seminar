### mnist.py
### Goal: You can build reusable program of FNN.
### Environment: tensorflow1.7.0, miniconda3-4.3.11, Python3.6.5, OSX10.13.4
### Reference: https://www.tensorflow.org/tutorials/deep_cnn

import os
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('learning_rate', 0.01,
                          """（初期）学習率""")
tf.app.flags.DEFINE_boolean('learning_rate_decay', True,
                            """学習率の減衰""")
tf.app.flags.DEFINE_float('num_epochs_per_decay', 350.0,
                          """学習率を減衰するインターバル""")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.1,
                          """学習率減衰係数""")
tf.app.flags.DEFINE_float('momentum', 0.9,
                          """慣性係数""")
tf.app.flags.DEFINE_boolean('use_nesterov', True,
                            """Nesterovの加速勾配降下法""")
tf.app.flags.DEFINE_integer('validation', 5000,
                            """バリデーションのサンプルサイズ""")

EXAMPLE_SIZE = 28*28    # 画像のサイズ
LABEL_SIZE = 10         # ラベルの数
TRAIN_SAMPLE_SIZE = 60000 - FLAGS.validation
VALID_SAMPLE_SIZE = FLAGS.validation
TEST_SAMPLE_SIZE = 10000


def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)

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


def affine(x, shape, seed, wd=0.004, stddev=0.1, bias=0.1):
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

    with tf.variable_scope('losses') as scope:
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

    with tf.variable_scope('accuracy') as scope:
        correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        correct_num = tf.reduce_sum(tf.cast(correct, tf.int32))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    return accuracy


def train(losses, global_step, train_sample_size):
    print('Use "mnist/train"')

    with tf.variable_scope('optimizer') as scope:
        # Variables that affect learning rate.
        num_batches_per_epoch = train_sample_size / FLAGS.batch_size
        decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay)

        # Decay the learning rate exponentially based on the number of steps.
        if FLAGS.learning_rate_decay:
            lr = tf.train.exponential_decay(FLAGS.learning_rate,
                                            global_step,
                                            decay_steps,
                                            FLAGS.learning_rate_decay_factor,
                                            staircase=True)
        else:
            lr = FLAGS.learning_rate
        tf.summary.scalar('learning_rate', lr)

        # Compute gradients.
        optimizer = tf.train.MomentumOptimizer(learning_rate=FLAGS.learning_rate,
                                               momentum=FLAGS.momentum,
                                               use_nesterov=FLAGS.use_nesterov)
        grads = optimizer.compute_gradients(losses['total_loss'])

        # Apply gradients.
        apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)

        with tf.control_dependencies([apply_gradient_op]):
            train_op = tf.no_op(name='train')

    print('optimizer                 : %s' % optimizer.get_name())
    print('learning_rate             : %f' % FLAGS.learning_rate)
    print('learning_rate_decay       : ' + str(FLAGS.learning_rate_decay))
    print('num_epochs_per_decay      : %f' % FLAGS.num_epochs_per_decay)
    print('learning_rate_decay_factor: %f' % FLAGS.learning_rate_decay_factor)
    print('momentum                  : %f' % FLAGS.momentum)
    print('use_nesterov              : ' + str(FLAGS.use_nesterov))

    return train_op
