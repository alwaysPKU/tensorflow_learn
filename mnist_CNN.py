"""
input + conv
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 准确度
def compute_accuacy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result


# 权重变量
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# 偏倚（偏置）
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 卷积层 x是输入，W是权重,strides 是步长(hengxiang he zongxiang buchang)
# padding "same" 是边缘0填充，所以大小不变
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# 池化层
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#定义placeholder存储输入数据
#每个图片是28*28=784像素，none表示不规定多少个sample，784表示规定输入大小
xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

x_image = tf.reshape(xs, [-1, 28, 28, 1])

# conv1 patch = 5*5  in size = 1(depth), out size = 32
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) #28*28*32
h_pool1 = max_pool_2x2(h_conv1)                          #14*14*32

# conv2 patch = 5*5  in size = 32(depth), out size = 64
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) #14*14*64
h_pool2 = max_pool_2x2(h_conv2)                          #7*7*64

# func layer1
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64]) # 把n_sample 7,7,64 改成 n_sample 7*7*64
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1)+b_fc1)
#加个dropout处理：
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# func layer2
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2)+b_fc2)

# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    # 现在开始train，每次只取100张图片，免得数据太多训练太慢。
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    # 每训练50次输出一下预测精度
    if i % 50 == 0:
        print(compute_accuacy(mnist.test.images[:1000], mnist.test.labels[:1000]))



