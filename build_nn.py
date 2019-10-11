import tensorflow as tf
import numpy as np


def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))  # 随机一个in_size行,out_size列的矩阵
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)  # 1行，out_size列，初始为推荐不为0，此处为0.1

    Wx_plus_b = tf.matmul(inputs, Weights) + biases

    if activation_function is None:
        outputs = Wx_plus_b  # 保持线性关系
    else:
        outputs = activation_function(Wx_plus_b)

    return outputs

x_data = np.linspace(-1,1,300)[:,np.newaxis]#300个例子，-1～1之间
noise = np.random.normal(0,0.05,x_data.shape)# 噪点，使数据更离散真实
y_data = np.square(x_data) - 0.5 + noise

#输入（出）层神经元数与x（y）参数个数一样
#此处设隐藏层神经元10个

#好处是 其他梯度下降方法时，run时只传入小部分数据更有效
xs = tf.placeholder(tf.float32,[None,1])#None意味着多少个例子都可以，1是参数个数
ys = tf.placeholder(tf.float32,[None,1])

#隐藏层
l1 = add_layer(xs,1,10,activation_function= tf.nn.relu)
#输出层
prediction = add_layer(l1,10,1,activation_function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                     reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i% 50 == 0:
        print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
#在不断提高预测准确性
