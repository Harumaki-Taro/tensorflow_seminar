### calc_graph.py
### 内容: 初歩的な計算グラフの使い方に慣れる。
### 動作確認: tensorflow1.7.0, miniconda3-4.3.11, Python3.6.5
### 参考文献: 新村拓哉, TensorFlowではじめるDeepLearning実装入門, 株式会社インプレス, 2018

import tensorflow as tf

###
### 計算グラフを構築する
###

# 定数と変数の箱の用意
a = tf.constant(3, name='const1')           # "定"数a
b = tf.Variable(0, name='val1')             # "変"数b

# aとbを足す演算の用意
add = tf.add(a, b)                          # "add = a + b"でもよい

# 変数bにaddの結果をassignに挿入
assign = tf.assign(b, add)

c = tf.placeholder(tf.int32, name='input')  # 入力c

# 挿入した結果とcを掛け算の用意
mul = tf.multiply(assign, c)

# 変数の初期化オペレーションの用意
init = tf.global_variables_initializer()

###
### 構築した計算グラフを用いて計算する
###

# 構築した計算グラフの実行
with tf.Session() as sess:
    # 初期化の実行
    sess.run(init)

    # 3回ループ
    for i in range(3):
        # 掛け算mulが実行されるまでの計算グラフを実行
        result = sess.run(mul, feed_dict={c:3})
        print(result)
