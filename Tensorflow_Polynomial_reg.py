import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

n_observations=100
fig, ax = plt.subplots(1,1)
x_data=np.linspace(-3, 3, n_observations)
y_data=np.sin(x_data)+np.random.uniform(-0.5, 0.5, n_observations)
x_data=np.reshape(x_data,[n_observations,1])
y_data=np.reshape(y_data,[n_observations,1])
plt.scatter(x_data, y_data)
plt.show()

X=tf.placeholder(dtype=tf.float32)
Y=tf.placeholder(dtype=tf.float32)

def model(x):
    y=tf.Variable(tf.random_normal(shape=[1], dtype=tf.float32), name='bias')
    for i in range(1,5):
        W=tf.Variable(tf.random_normal(shape=[1], dtype=tf.float32), name='weight_%d' %i)
        y=tf.add(tf.multiply(tf.pow(x,i),W),y)
    return y

logits=model(X)
loss=tf.reduce_sum(tf.pow(logits-Y, 2)/ (n_observations-1))
optimizer=tf.train.GradientDescentOptimizer(0.01)
train_step=optimizer.minimize(loss)

init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    n_epochs=1000
    for i in range(n_epochs):
        for (x,y) in zip(x_data, y_data):
            sess.run(train_step, feed_dict={X:x, Y:y})
        if i==0 or (i+1)%50==0 or i<5:
            print('Epoch: ',(i+1),' Loss: ',sess.run(loss, feed_dict={X:x_data, Y:y_data}))
