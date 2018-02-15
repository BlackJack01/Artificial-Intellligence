from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('/tmp/data/', one_hot =True)

import tensorflow as tf
from tensorflow.contrib import rnn

## Model's Hyperparameters
learning_rate=0.001
batch_size=128
training_step=10000
display_step=500

n_input=28
timesteps=28
n_hidden=128
n_output=10

X=tf.placeholder(shape=[None, timesteps, n_input], dtype=tf.float32)
Y=tf.placeholder(shape=[None,n_output], dtype=tf.float32)

## Weights and Biases
W={'out':tf.Variable(tf.random_normal(shape=[n_hidden,n_output]))}
b={'out':tf.Variable(tf.random_normal(shape=[1,n_output]))}

def model(x):
    x=tf.unstack(x, timesteps, 1)
    lstm= rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    outputs, states= rnn.static_rnn(lstm, x, dtype=tf.float32)
    return tf.add(tf.matmul(outputs[-1], W['out']), b['out'])

logits=model(X)

loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op=optimizer.minimize(loss)

correct_predictions=tf.equal(tf.argmax(logits,1), tf.argmax(Y, 1))
accuracy=tf.reduce_mean(tf.cast(correct_predictions, dtype=tf.float32))

init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(training_step):
        x_batch, y_batch = mnist.train.next_batch(batch_size)
        x_batch=x_batch.reshape((batch_size,timesteps,n_input))
        sess.run(train_op, feed_dict={X:x_batch, Y:y_batch})
        if (i+1)%display_step==0 or i==0:
            acc=sess.run(accuracy, feed_dict={X:x_batch, Y:y_batch})
            print("Step: ",(i+1)," accuracy: ",acc)
