import tensorflow as tf
import numpy as np


# 给出input数据（x，y），x为随机，y为x的一元函数
# 实现的是通过训练，初始w和b是随机数，通过减少训练结果y与实际y的误差，不断优化w和b取值
# 最终得到相近的w和b

#create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3

#create tensoflow struture start ##

Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))#一维，范围-1～1
biases = tf.Variable(tf.zeros([1]))

y = Weights*x_data + biases

loss = tf.reduce_mean(tf.square(y-y_data))

optimizer = tf.train.GradientDescentOptimizer(0.5)#学习效率
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

#create tensoflow struture end ##

sess = tf.Session()
sess.run(init)  #!!!important

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step,sess.run(Weights),sess.run(biases))

