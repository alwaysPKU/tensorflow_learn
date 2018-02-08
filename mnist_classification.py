import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 添加层函数，返回这层layer的输出
def add_layer(inpusts, in_size, out_size, n_layer, activation_function = None):
    layer_name = 'layer%s'%n_layer
    with tf.name_scope(layer_name):#起名字为了在tensorboard显示
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            tf.summary.histogram(layer_name + '/weights', Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size])) + 0.1
            tf.summary.histogram(layer_name + '/biase', biases)
        with tf.name_scope('Wx_plux_b'):
            Wx_plus_b = tf.matmul(inpusts, Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        tf.summary.histogram(layer_name + '/outputs', outputs)
    return outputs

#定义placeholder存储输入数据
#每个图片是28*28=784像素，none表示不规定多少个sample，784表示规定输入大小
xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])#输出是十纬度的向量，表示0，1，2，3，4，5，6，7，8，9

#添加输出层
prediction = add_layer(xs, 784, 10, 1,activation_function=tf.nn.softmax)

#定义损失函数：loss函数（即最优化目标函数）选用交叉熵函数。
# 交叉熵用来衡量预测值和真实值的相似程度，如果完全相同，它们的交叉熵等于零。
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction), reduction_indices=[1]))

#训练:train方法（最优化算法）采用梯度下降法。
train_step = tf.train.GradientDescentOptimizer(0.095).minimize(cross_entropy)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 定义精度函数
def compute_accuacy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs:v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs:v_xs, ys:v_ys})
    return result



for i in range(1000):
    # 现在开始train，每次只取100张图片，免得数据太多训练太慢。
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs:batch_xs, ys:batch_ys})
    # 每训练50次输出一下预测精度
    if i%50==0:
        print(compute_accuacy(mnist.test.images, mnist.test.labels))


