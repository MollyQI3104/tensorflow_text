import tensorflow as tf


matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],
                       [2]])

product = tf.matmul(matrix1,matrix2)  #matrix multifly np.dot(m1,m2)

#method1

# sess = tf.Session()
# result = sess.run(product)
#
# print(result)
# sess.close()

#method2
# 运行到最后sess自动关闭
with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)

