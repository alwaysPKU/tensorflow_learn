# 测试session用法

import tensorflow as tf

# 一行两列
matrix1 = tf.constant([[3, 3]])
# 两行一列
matrix2 = tf.constant([[2], [2]])
# 矩阵相乘 类似numpy的dot np.dot(1,2)
product = tf.matmul(matrix1, matrix2)

sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()