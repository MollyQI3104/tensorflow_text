import tensorflow as tf

# 定义了variable一定要init（初始化）
# init = tf.initialize_all_variables()
# 定义完sess一定要run init（激活）

state = tf.Variable(0, name = 'counter') #set value and name
# print(state.name)

one = tf.constant(1)

new_value = tf.add(state , one)
update = tf.assign(state, new_value)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))