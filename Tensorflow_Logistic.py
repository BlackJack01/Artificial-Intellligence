import tensorflow as tf
import numpy

## Model 
A=tf.Variable(tf.random_normal(shape=[5,1]))
b=tf.Variable(tf.random_normal(shape=[1,1]))
x_data=tf.placeholder(shape=[None,5], dtype=tf.float32)
y_data=tf.placeholder(shape=[None,1], dtype=tf.float32)
model=tf.add(tf.matmul(x_data,A),b)

## Data
train_X=numpy.array([[1,2,3,4,5],[5,4,3,2,1],[1,2,3,4,5],[5,4,3,2,2],[1,2,3,4,5],[5,4,3,2,1],[1,2,3,4,5],[4,4,3,2,1],[1,2,3,4,5],[5,4,3,2,1]])
train_Y=numpy.array([[1],[0],[1],[0],[1],[0],[1],[0],[1],[0]])

## Loss Calculation
soft_max=tf.nn.sigmoid_cross_entropy_with_logits(logits=model,labels=y_data)
loss=tf.reduce_mean(soft_max)

## Prediction and Accuracy Calculations
pred=tf.nn.sigmoid(model)
prediction=tf.round(pred)
prediction_correct=tf.cast(tf.equal(prediction,y_data), tf.float32)
accuracy=tf.reduce_mean(prediction_correct)

## Optimizer
my_opt=tf.train.GradientDescentOptimizer(0.0025)
train_step=my_opt.minimize(loss)

## Model Run
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        sess.run(train_step, feed_dict={x_data:train_X, y_data:train_Y})
        var=sess.run(accuracy, feed_dict={x_data:train_X, y_data:train_Y})
        if i%100==0:
            print(var)
