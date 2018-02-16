import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

n_observations=100
x_data=np.linspace(-3, 3, n_observations)
#print(x_data.size)
y_data=np.sin(x_data)+np.random.uniform(-0.5, 0.5, n_observations)
x_data=np.reshape(x_data, [100,1])
y_data=np.reshape(y_data, [100,1])
#print(y_data.size)
plt.scatter(x_data, y_data)
plt.show()

X=tf.placeholder(shape=[None,1], dtype=tf.float32)
Y=tf.placeholder(shape=[None,1], dtype=tf.float32)

W=tf.Variable(tf.random_normal(shape=[1], dtype=tf.float32))
b=tf.Variable(tf.random_normal(shape=[1], dtype=tf.float32))

model=tf.add(tf.multiply(X,W),b)

loss=tf.reduce_sum(tf.pow(model-Y, 2)/(n_observations-1))
optimizer=tf.train.GradientDescentOptimizer(0.01)
train_step=optimizer.minimize(loss)
init = tf.global_variables_initializer()

n_epochs=1000
with tf.Session() as sess:
    sess.run(init)
    for i in range(n_epochs):
        sess.run(train_step, feed_dict={X:x_data, Y:y_data})
        if i==1 or (i+1)%50==0:
            print('Epoch: ',(i+1),' Loss = ', sess.run(loss, feed_dict={X:x_data, Y:y_data}))
