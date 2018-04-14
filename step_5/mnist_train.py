### mnist_train.py
### Goal: You can build reusable program of TensorFlow.
### Environment: tensorflow1.7.0, miniconda3-4.3.11, Python3.6.5, OSX10.13.4
### Reference: https://www.tensorflow.org/tutorials/deep_cnn

import tensorflow as tf
import datetime
import time
import os
import platform
import socket
import pkg_resources
from git import *

import util
import mnist
import mnist_full_connect_4


FLAGS = tf.app.flags.FLAGS
start_time = datetime.datetime.now()
start_time_str = start_time.strftime("%Y%m%d%H%M%S")
tf.app.flags.DEFINE_string('name', start_time_str,
                           """実験名""")
tf.app.flags.DEFINE_integer('max_steps', 5000,
                            """学習回数""")
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """バッチサイズ""")
tf.app.flags.DEFINE_integer('log_interval', 50,
                            """ログを出力する間隔""")
tf.app.flags.DEFINE_integer('max_to_keep', 10,
                            """モデル保存数の上限""")
tf.app.flags.DEFINE_integer('save_interval', 500,
                            """モデルの保存の間隔""")
tf.app.flags.DEFINE_boolean('write_meta_graph', False,
                            """メタファイルの書き込み""")


def model(graph, images, labels):
    with graph.as_default():
        with tf.device('/cpu:0'):
            global_step = tf.train.get_or_create_global_step()

        logits = mnist_full_connect_4.inference(inputs=images)
        y = tf.nn.softmax(logits)

        losses = util.loss(logits, labels)
        accuracy = util.accuracy(logits, labels)
        train_op = util.train(losses=losses,
                              global_step=global_step,
                              train_sample_size=mnist.TRAIN_SAMPLE_SIZE)

        saver = tf.train.Saver(max_to_keep=FLAGS.max_to_keep)
        summary_op = tf.summary.merge_all()

        train_ops = {'global_step': global_step,
                     'train'      : train_op,
                     'summary'    : summary_op}
        if len(tf.get_collection('regularizations')) > 0:
            valid_ops = {'x_entropy'  : losses['cross_entropy'],
                         'regs'       : losses['regularizations'],
                         'total_loss' : losses['total_loss'],
                         'acc'        : accuracy}
        else:
            valid_ops = {'x_entropy'  : losses['cross_entropy'],
                         'total_loss' : losses['total_loss'],
                         'acc'        : accuracy}

    return train_ops, valid_ops, graph, global_step, saver


def train():
    mnist_data = mnist.read_data_sets()
    with tf.Graph().as_default() as graph:
        with tf.variable_scope('mnist') as scope:
            with tf.device('/cpu:0'):
                images = tf.placeholder(tf.float32, [None, mnist.EXAMPLE_SIZE], name='images')
                labels = tf.placeholder(tf.float32, [None, mnist.LABEL_SIZE], name='labels')

    train_ops, valid_ops, graph, global_step, saver = model(graph, images, labels)

    log_dir = 'ckpt'

    with tf.Session(graph=graph) as sess:
        writer = tf.summary.FileWriter(log_dir, sess.graph)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        step = sess.run(global_step)

        test_images = mnist_data.test.images
        test_labels = mnist_data.test.labels
        test_feed_dict = {images:test_images, labels:test_labels}

        if len(tf.get_collection('regularizations')) > 0:
            print('# step\tx_entropy\tregs\ttotal_loss\tacc\tmsec/100examples\texamples/sec\ttime')
        else:
            print('# step\tx_entropy\ttotal_loss\tacc\tmsec/100examples\texamples/sec\ttime')

        while step < FLAGS.max_steps:
            start_time = time.time()
            train_images, train_labels = mnist_data.train.next_batch(FLAGS.batch_size)
            result = sess.run(train_ops,
                              feed_dict={images:train_images, labels:train_labels})
            step = result['global_step']
            writer.add_summary(result['summary'], step)
            duration = time.time() - start_time

            if step % FLAGS.log_interval == 0:
                validate(sess, valid_ops, test_feed_dict, step, duration)

            if step % FLAGS.save_interval == 0:
                saver.save(sess,
                           'ckpt/'+FLAGS.name+'_',
                           global_step=step,
                           write_meta_graph=FLAGS.write_meta_graph)


def validate(sess, valid_ops, test_feed_dict, step, duration):
    # accuracyを計算
    result = sess.run(valid_ops,
                      feed_dict=test_feed_dict)
    now = datetime.datetime.now()

    if len(tf.get_collection('regularizations')) > 0:
        print('%d\t%.4f\t%.4f\t%.4f\t%.3f\t%.1f\t%.1f\t%s'
                % (step,
                   result['x_entropy'],
                   result['regs'],
                   result['total_loss'],
                   result['acc'],
                   duration/FLAGS.batch_size*100*1000,
                   FLAGS.batch_size/duration,
                   now.strftime('%Y/%m/%d %H:%M:%S')))

    else:
        print('%d\t%.4f\t%.4f\t%.3f\t%.1f\t%.1f\t%s'
                % (step,
                   result['x_entropy'],
                   result['total_loss'],
                   result['acc'],
                   duration/FLAGS.batch_size*100*1000,
                   FLAGS.batch_size/duration,
                   now.strftime('%Y/%m/%d %H:%M:%S')))


def write_header():
    modules = []
    for dist in pkg_resources.working_set:
        modules.append([dist.project_name, dist.version])
    print('modules      : ' + str(modules))

    repo = Repo(os.getcwd()+'/../')
    head = repo.head
    master = head.reference
    log = master.log()
    print('git          : ' + str(log[-1])[:-1])
    print('os           : %s' % platform.system())
    print('host name    : %s' % socket.gethostname())
    print('path         : %r' % os.path.abspath(__file__))
    print('date         : ' + start_time.strftime('%Y/%m/%d %H:%M:%S'))
    print('max_steps    : %d' % FLAGS.max_steps)
    print('batch_size   : %d' % FLAGS.batch_size)
    print('log_interval : %d' % FLAGS.log_interval)
    print('max_to_keep  : %d' % FLAGS.max_to_keep)
    print('save_interval: %d' % FLAGS.save_interval)


def main(argv=None):
    write_header()
    train()


if __name__ == '__main__':
    tf.app.run()
