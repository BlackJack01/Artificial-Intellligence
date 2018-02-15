## Data import 
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf

## Model's Hyperparameters
learning_rate=0.1
num_steps=500
batch_size=128
display_steps=50

n_input=784
n_hidden1=128
n_hidden2=128
n_output=10

## Inputs
X=tf.placeholder(shape=[None,n_input], dtype=tf.float32)
Y=tf.placeholder(shape=[None,n_output], dtype=tf.float32)

## Initial weights and biases for all layers
''''
W={
    'h1':tf.Variable(tf.random_normal(shape=[n_input,n_hidden1])),
    'h2':tf.Variable(tf.random_normal(shape=[n_hidden1,n_hidden2])),
    'out':tf.Variable(tf.random_normal(shape=[n_hidden2,n_output]))
}
b={
    'h1':tf.Variable(tf.random_normal(shape=[n_hidden1])),
    'h2':tf.Variable(tf.random_normal(shape=[n_hidden2])),
    'out':tf.Variable(tf.random_normal(shape=[n_output]))
}
''''
## Neural Model Definition
def model(x):
    #layer1=tf.add(tf.matmul(x,W['h1']),b['h1'])
    #layer2=tf.add(tf.matmul(layer1,W['h2']),b['h2'])
    #out_layer=tf.add(tf.matmul(layer2,W['out']),b['out'])
    
    layer1=tf.layers.dense(x,n_hidden1)
    layer2=tf.layers.dense(layer1, n_hidden2)
    out_layer=tf.layers.dense(layer2,n_output)
    return out_layer

logits=model(X)

loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
opt=tf.train.AdamOptimizer(learning_rate=learning_rate)
train_step=opt.minimize(loss)

correct_prediction=tf.equal(tf.argmax(logits,1), tf.argmax(Y,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(num_steps):
        x_batch, y_batch = mnist.train.next_batch(batch_size)
        sess.run(train_step, feed_dict={X:x_batch, Y:y_batch})
        if (i+1)%display_steps ==0 or (i+1)==1:
            acc=sess.run(accuracy, feed_dict={X:x_batch, Y:y_batch})
            print("Step ",(i+1)," Accuracy = ",acc)
    
    print("Testing Accuracy = ",sess.run(accuracy, feed_dict={X:mnist.test.images, Y:mnist.test.labels}))
