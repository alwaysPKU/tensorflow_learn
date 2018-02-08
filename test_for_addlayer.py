import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt

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

x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

# 利用占位符定义我们所需的神经网络的输入。 tf.placeholder()就是代表占位符，
# 这里的None代表无论输入有多少都可以，因为输入只有一个特征，所以这里是1。
with tf.name_scope('inputs'):#定义input层显示
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')


# 接下来，我们就可以开始定义神经层了。
# 通常神经层都包括输入层、隐藏层和输出层。
# 这里的输入层只有一个属性， 所以我们就只有一个输入；
# 隐藏层我们可以自己假设，这里我们假设隐藏层有10个神经元；
# 输出层和输入层的结构是一样的，所以我们的输出层也是只有一层。
# 所以，我们构建的是——输入层1个、隐藏层10个、输出层1个的神经网络。

# 下面，我们开始定义隐藏层,利用之前的add_layer()函数，
# 这里使用 Tensorflow 自带的激励函数tf.nn.relu。
# 三层神经，输入层（1个神经元），隐藏层（10神经元），输出层（1个神经元）
l1 = add_layer(xs, 1, 10, n_layer=1, activation_function=tf.nn.relu)

# 接着，定义输出层。此时的输入就是隐藏层的输出——l1，输入有10层（隐藏层的输出层），输出有1层。
prediction = add_layer(l1, 10, 1, n_layer=2, activation_function=None)

# 计算预测值prediction和真实值的误差，对二者差的平方求和再取平均。
with tf.name_scope('loss'):#定义损失函数
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
    tf.summary.scalar('loss', loss)

# 接下来，是很关键的一步，如何让机器学习提升它的准确率。
# tf.train.GradientDescentOptimizer()中的值通常都小于1，
# 这里取的是0.1，代表以0.1的效率来最小化误差loss。
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 初始化
init = tf.global_variables_initializer()
# 定义Session，并用 Session 来执行 init 初始化步骤。
# （注意：在tensorflow中，只有session.run()才会执行我们定义的运算。）
sess = tf.Session()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("logs", sess.graph)
sess.run(init)


# 下面，让机器开始学习。
#
# 比如这里，我们让机器学习1000次。机器学习的内容是train_step,
# 用 Session 来 run 每一次 training 的数据，逐步提升神经网络的预测准确性。
# (注意：当运算要用到placeholder时，就需要feed_dict这个字典来指定输入。)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)
plt.ion()#画完不终止
plt.show()

for i in range(1000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    # if i%50 == 0:
    #     print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
    if i % 50 ==0:
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = sess.run(prediction, feed_dict={xs: x_data})
        lines = ax.plot(x_data, prediction_value, 'r-', lw=3)
        plt.pause(0.1)
        rs = sess.run(merged, feed_dict={xs: x_data, ys: y_data})
        writer.add_summary(rs, i)