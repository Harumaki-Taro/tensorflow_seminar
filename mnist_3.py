### mnist_3.py
### 目的: 研究用のプログラムを書けるようになる。
### 動作確認: tensorflow1.7.0, miniconda3-4.3.11, Python3.6.5, OSX10.13.4
### 参考文献: 新村拓哉, TensorFlowではじめるDeepLearning実装入門, 株式会社インプレス, 2018

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import time


EXAMPLE_SIZE = 28*28    # 画像のサイズ
LABEL_SIZE = 10         # ラベルの数

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('h1_unit_num', 300,
                            """隠れ層1のユニット数""")
tf.app.flags.DEFINE_integer('h2_unit_num', 64,
                            """隠れ層2のユニット数""")
tf.app.flags.DEFINE_integer('max_steps', 5000,
                            """学習回数""")
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """バッチサイズ""")
tf.app.flags.DEFINE_integer('log_interval', 50,
                            """ログを出力するインターバル""")
tf.app.flags.DEFINE_integer('seed', 1234,
                            """random seed""")

###
### mnistデータセットの準備（tensorflow的に本来はここも計算グラフに入るはず）
###

# mnistデータセットを格納したオブジェクトを呼び出す
mnist = input_data.read_data_sets('data/', one_hot=True)

###
### 計算グラフ(ニューラルネット)を構築する
###

# 入力層の用意
with tf.name_scope('input_layer') as scope:
    x_0 = tf.placeholder(tf.float32, [None, EXAMPLE_SIZE], name=scope)

# 隠れ層1の用意
with tf.name_scope('hidden_layer_1') as scope:
    w_1 = tf.Variable(tf.truncated_normal([EXAMPLE_SIZE, FLAGS.h1_unit_num],
                      mean=0.0, stddev=0.1, seed=FLAGS.seed), name='weight')
    b_1 = tf.Variable(tf.zeros([FLAGS.h1_unit_num]), name='bias')
    x_1 = tf.nn.relu(tf.matmul(x_0, w_1) + b_1)
    tf.summary.histogram(name=w_1.name, values=w_1)
    tf.summary.histogram(name=b_1.name, values=b_1)
    tf.summary.histogram(name=x_1.name, values=x_1)

# 隠れ層2の用意
with tf.name_scope('hidden_layer_2') as scope:
    w_2 = tf.Variable(tf.truncated_normal([FLAGS.h1_unit_num, FLAGS.h2_unit_num],
                      mean=0.0, stddev=0.1, seed=FLAGS.seed+10), name='weight')
    b_2 = tf.Variable(tf.zeros([FLAGS.h2_unit_num]), name='bias')
    x_2 = tf.nn.relu(tf.matmul(x_1, w_2) + b_2)
    tf.summary.histogram(name=w_2.name, values=w_2)
    tf.summary.histogram(name=b_2.name, values=b_2)
    tf.summary.histogram(name=x_2.name, values=x_2)

# 出力層の用意
with tf.name_scope('output_layer') as scope:
    w_3 = tf.Variable(tf.truncated_normal([FLAGS.h2_unit_num, LABEL_SIZE],
                      mean=0.0, stddev=0.1, seed=FLAGS.seed+20), name='weight')
    b_3 = tf.Variable(tf.zeros([LABEL_SIZE]), name='bias')
    logit = tf.add(tf.matmul(x_2, w_3), b_3, name='logit')
    tf.summary.histogram(name=w_3.name, values=w_3)
    tf.summary.histogram(name=b_3.name, values=b_3)
    tf.summary.histogram(name=logit.name, values=logit)

# softmax
y = tf.nn.softmax(logit)

# 評価値の用意
with tf.name_scope('evaluation') as scope:
    # 損失関数の用意
    t = tf.placeholder(tf.float32, [None, LABEL_SIZE])
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=t, logits=logit), name='loss')
    tf.summary.scalar('loss', loss)                             # lossのログを取得するように設定
    # 評価値の用意
    correct = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))        # 正答したかどうかのboolの配列
    correct_num = tf.reduce_sum(tf.cast(correct, tf.int32))     # 正答数
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))     # 正答率
    tf.summary.scalar('accuracy', accuracy)                     # accuracyのログを取得するように設定

# トレーニングオペレーションの用意（Nesterov Momentum(Sutskever et al., 2013)）
optimizer = tf.train.MomentumOptimizer(learning_rate=0.01,
                                       momentum=0.9,
                                       use_nesterov=True)
train_op = optimizer.minimize(loss)

# 初期化オペレーションの用意
init = tf.global_variables_initializer()


###
### 構築した計算グラフを用いて計算する
###

# 構築した計算グラフの実行
with tf.Session() as sess:
    # 初期化の実行
    sess.run(init)

    # テストデータをロード
    # テスト用の全ての画像データを取得
    test_images = mnist.test.images
    #テスト用の全ての教師データを取得
    test_labels = mnist.test.labels

    for step in range(FLAGS.max_steps):
        # 訓練用の入力データ、教師データを取得
        #（ミニバッチ数を設定することでsessionの中で使うと新しいデータを50個ずつランダムにポップしてくれる）
        train_images, train_labels = mnist.train.next_batch(FLAGS.batch_size)

        # train_opを実行
        start_time = time.time()
        sess.run(train_op, feed_dict={x_0:train_images, t:train_labels})
        duration = time.time() - start_time

        if step % FLAGS.log_interval == 0:
            # accuracyを計算
            loss_val, acc_val = sess.run([loss, accuracy],
                                         feed_dict={x_0:test_images, t:test_labels})
            print('step %d: loss = %.4f acc = %.3f  %.1f[msec/step]' % (step, loss_val, acc_val, duration*1000))
