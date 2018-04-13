### mnist_1.py
### 目的: mnistを学習する最低限のニューラルネットワークを作れるようになる。
### 動作確認: tensorflow1.7.0, miniconda3-4.3.11, Python3.6.5, OSX10.13.4
### 参考文献: 新村拓哉, TensorFlowではじめるDeepLearning実装入門, 株式会社インプレス, 2018

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


###
### mnistデータセットの準備（tensorflow的に本来はここも計算グラフに入るはず）
###

# mnistデータセットを格納したオブジェクトを呼び出す
mnist = input_data.read_data_sets('data/', one_hot=True)

###
### 計算グラフ(ニューラルネット)を構築する
###

# 入力層の用意(mnist画像のサイズが28*28=784次元のベクトル)
x_0 = tf.placeholder(tf.float32, [None, 28*28])

# 隠れ層の用意
w_1 = tf.Variable(tf.random_normal([28*28, 64], mean=0.0, stddev=0.1), name='w_1')
b_1 = tf.Variable(tf.zeros([64]), name='b_1')
x_1 = tf.nn.relu(tf.matmul(x_0, w_1) + b_1)

# 出力層の用意
w_2 = tf.Variable(tf.random_normal([64, 10], mean=0.0, stddev=0.1), name='w_2')
b_2 = tf.Variable(tf.zeros([10]), name='b_2')
logit = tf.matmul(x_1, w_2) + b_2
y = tf.nn.softmax(logit)

# 損失関数の用意
t = tf.placeholder(tf.float32, [None, 10])
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=t, logits=logit), name='loss')

# トレーニングオペレーションの用意(確率的勾配降下法(SGD))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train_op = optimizer.minimize(loss)

# 評価値の用意
correct = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))        # 正答したかどうかのboolの配列
correct_num = tf.reduce_sum(tf.cast(correct, tf.int32))     # 正答数
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))     # 正答率

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

    for step in range(1000):
        # 訓練用の入力データ、教師データを取得
        #（ミニバッチ数を設定することでsessionの中で使うと新しいデータを50個ずつランダムにポップしてくれる）
        train_images, train_labels = mnist.train.next_batch(50)

        # train_opを実行
        sess.run(train_op, feed_dict={x_0:train_images, t:train_labels})

        if step % 10 == 0:
            # accuracyを計算
            correct_val, acc_val = sess.run([correct_num, accuracy], feed_dict={x_0:test_images, t:test_labels})
            print('step %d: correct_num = %d/10000  acc = %.2f' % (step, correct_val, acc_val))
