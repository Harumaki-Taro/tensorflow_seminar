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

import mnist
import mnist_full_connect_4


EXAMPLE_SIZE = 28*28    # 画像のサイズ
LABEL_SIZE = 10         # ラベルの数

FLAGS = tf.app.flags.FLAGS
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


def train():
    # mnistデータセットを格納したオブジェクトを呼び出す
    mnist_data = mnist.read_data_sets()

    images = tf.placeholder(tf.float32, [None, EXAMPLE_SIZE], name='input_layer')
    labels = tf.placeholder(tf.float32, [None, LABEL_SIZE])

    # Neural network
    with tf.name_scope('neural_network'):
        logits = mnist_full_connect_4.inference(inputs=images)
        y = tf.nn.softmax(logits)

    losses = mnist.loss(logits, labels)          # 損失関数
    accuracy = mnist.accuracy(logits, labels)    # 正答率
    train_op = mnist.train(losses)          # トレーニングオペレーション

    # モデルの保存オペレーション
    saver = tf.train.Saver()

    # 初期化オペレーションの用意
    init = tf.global_variables_initializer()

    # 構築した計算グラフの実行
    with tf.Session() as sess:
        # 初期化の実行
        sess.run(init)

        # テストデータをロード
        test_images = mnist_data.test.images
        test_labels = mnist_data.test.labels
        test_feed_dict = {images:test_images, labels:test_labels}

        if len(tf.get_collection('regularizations')) > 0:
            print('# step\tx_entropy\tregs\ttotal_loss\tacc\tmsec/100examples\texamples/sec\ttime')
        else:
            print('# step\tx_entropy\ttotal_loss\tacc\tmsec/100examples\texamples/sec\ttime')

        for step in range(FLAGS.max_steps):
            # 訓練用の入力データ、教師データを取得
            train_images, train_labels = mnist_data.train.next_batch(FLAGS.batch_size)

            # train_opを実行
            start_time = time.time()
            sess.run(train_op, feed_dict={images:train_images, labels:train_labels})
            duration = time.time() - start_time

            if step % FLAGS.log_interval == 0:
                validate(sess, losses, accuracy, test_feed_dict, step, duration)

            if step % FLAGS.save_interval == 0:
                saver.save(sess, 'ckpt/model', global_step=step)


def validate(sess, losses, accuracy, test_feed_dict, step, duration):
    # accuracyを計算
    if len(tf.get_collection('regularizations')) > 0:
        result = sess.run({'x_entropy'  : losses['cross_entropy'],
                           'regs'       : losses['regularizations'],
                           'total_loss' : losses['total_loss'],
                           'acc'        : accuracy},
                           feed_dict=test_feed_dict)
        now = datetime.datetime.now()
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
        result = sess.run({'x_entropy'  : losses['cross_entropy'],
                           'total_loss' : losses['total_loss'],
                           'acc'        : accuracy},
                           feed_dict=test_feed_dict)
        now = datetime.datetime.now()
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
    print('date         : ' + datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
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
