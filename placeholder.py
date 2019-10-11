import tensorflow as tf

# placeholder意味着在sess.run()的时候再给值
# feed_dict = {key:value}
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1,input2)

with tf.Session() as sess:
    print(sess.run(output,feed_dict={input1:[7.],input2:[2.]}))