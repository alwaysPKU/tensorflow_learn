import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 可视化准备
plot_data = {'batch_size': [], 'loss': []}
def moving_average(a, w=10):
    if len(a) < w:
        return a[:]
    return [val if idx < w else sum(a[(idx-w): idx])/w for idx, val in enumerate(a)]

# 准备数据
train_X = np.linspace(-1, 1, 100)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3
# plt.plot(train_X, train_Y, 'b*', label='Original data')
# plt.legend()
# plt.show()
# 搭建模型
## 占位符
X = tf.placeholder('float')
Y = tf.placeholder('float')
## 模型参数
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.zeros([1]), name='bias')
## 前向结构
z = tf.multiply(X, W) + b
tf.summary.histogram('z', z)
## 反向优化
cost = tf.reduce_mean(tf.square(Y - z))
tf.summary.scalar('loss_function', cost)
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
# 训练
init = tf.global_variables_initializer()
training_epochs = 20
display_step = 2
# saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    merged_summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter('log/mnist_with_summaries', sess.graph)
    # plot_data = {'batch_size': [], 'loss': []}
    for epoch in range(training_epochs):
        # sess.run(optimizer, feed_dict={X: train_X, Y: train_Y})

        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})
        if epoch % display_step == 0:
            loss = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
            print('Epoch', epoch+1, 'cost=', loss, 'W=', sess.run(W), 'b=', sess.run(b))
            if not (loss == 'NA'):
                plot_data['batch_size'].append(epoch)
                plot_data['loss'].append(loss)
        summary_str = sess.run(merged_summary_op, feed_dict={X: train_X, Y: train_Y})
        summary_writer.add_summary(summary_str, epoch)
    print('finished')
    print('cost=', sess.run(cost, feed_dict={X: train_X, Y: train_Y}), 'W=', sess.run(W), 'b=', sess.run(b))

    # saver.save(sess, './liner_regression_2x')
    plt.plot(train_X, train_Y, 'b*', label='Original data')
    plt.plot(train_X, sess.run(W)*train_X + sess.run(b), 'r', label='fitted line')
    plt.legend()
    plt.show()
plot_data['avgloss'] = moving_average(plot_data['loss'])
plt.figure(1)
plt.subplot(312)
plt.plot(plot_data['batch_size'], plot_data['avgloss'], 'r--')
plt.xlabel('Loss')
plt.ylabel('Minibatch run vs. Training loss')
plt.show()

