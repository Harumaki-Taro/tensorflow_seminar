### mnist_full_connect_4.py
### Goal: You can build reusable program of FNN.
### Environment: tensorflow1.7.0, miniconda3-4.3.11, Python3.6.5, OSX10.13.4
### Reference: https://www.tensorflow.org/tutorials/deep_cnn

import tensorflow as tf
import util
import mnist

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('h1_unit_num', 300,
                            """隠れ層1のユニット数""")
tf.app.flags.DEFINE_integer('h2_unit_num', 64,
                            """隠れ層2のユニット数""")
tf.app.flags.DEFINE_integer('seed', 1234,
                            """random seed""")
tf.app.flags.DEFINE_integer('seed_add', 7,
                            """random seedの間隔""")


def inference(inputs):
    print('***START building neural network***')
    with tf.name_scope('neural_network'):
        # 隠れ層1
        with tf.variable_scope('full_1') as scope:
            print('<layer 1: %s>' % scope.name)
            affine = util.affine(inputs,
                                 [mnist.EXAMPLE_SIZE, FLAGS.h1_unit_num],
                                 FLAGS.seed)
            full_1 = util.activate(tf.nn.relu, affine)

        # 隠れ層2
        with tf.variable_scope('full_2') as scope:
            print('<layer 2: %s>' % scope.name)
            affine = util.affine(full_1,
                                 [FLAGS.h1_unit_num, FLAGS.h2_unit_num],
                                 FLAGS.seed+FLAGS.seed_add)
            full_2 = util.activate(tf.nn.relu, affine)

        # 出力層
        with tf.variable_scope('logits') as scope:
            print('<layer 3: %s>' % scope.name)
            logits = util.affine(full_2,
                                 [FLAGS.h2_unit_num, mnist.LABEL_SIZE],
                                 FLAGS.seed+FLAGS.seed_add*2)

    print('***DONE building neural network***')

    return logits
