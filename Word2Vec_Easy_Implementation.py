import numpy as np
import tensorflow as tf
corpus_raw = 'He is the king . The king is royal . She is the royal  queen '

corpus_raw=corpus_raw.lower()
words=list(word for word in corpus_raw.split() if not word == '.')

vocab=set(words)
word2int={}
int2word={}
for i,word in enumerate(vocab):
    word2int[word]=i
    int2word[i]=word

raw_sentences=corpus_raw.split('.')
sentences=list(sentence.split() for sentence in raw_sentences) 

data=[]
window_size=2
for sentence in sentences:
    for index, word in enumerate(sentence):
        for new_word in sentence[max(0,index-window_size):min(index+window_size, len(sentence))+1]:
            if new_word!=word:
                data.append([word, new_word])
 
 print(data)
 
 def one_hot(data_point_index, vocab_size):
    temp=np.zeros(vocab_size)
    temp[data_point_index]=1
    return temp 
    
 x_train=[]
y_train=[]
for word_pair in data:
    x_train.append(one_hot(word2int[word_pair[0]], len(vocab)))
    y_train.append(one_hot(word2int[word_pair[1]], len(vocab)))

x_train=np.asarray(x_train)
y_train=np.asarray(y_train)

vocab_size=len(vocab)

X = tf.placeholder(shape=[None, vocab_size], dtype=tf.float32)
Y = tf.placeholder(shape=[None, vocab_size], dtype=tf.float32)

embedding_dims=5
W={
    'h':tf.Variable(tf.random_normal([vocab_size,embedding_dims])),
    'out':tf.Variable(tf.random_normal([embedding_dims,vocab_size]))
}
b={
    'h':tf.Variable(tf.random_normal([embedding_dims])),
    'out':tf.Variable(tf.random_normal([vocab_size]))
}

def model(x):
    h=tf.add(tf.matmul(x,W['h']),b['h'])
    out=tf.nn.softmax(tf.add(tf.matmul(h,W['out']),b['out']))
    return out
    
prediction=model(X)
init=tf.global_variables_initializer()
loss=tf.reduce_mean(-tf.reduce_sum(Y*tf.log(prediction), reduction_indices=[1]))
optimizer=tf.train.GradientDescentOptimizer(0.1)
train_step=optimizer.minimize(loss)

n_iterations=10000
sess=tf.Session()
sess.run(init)
for i in range(n_iterations):
    sess.run(train_step, feed_dict={X:x_train, Y:y_train})
    if (i+1)%100==0:
        print("Iteration: ",(i+1)," Loss: ",sess.run(loss, feed_dict={X:x_train, Y:y_train}))

vectors=sess.run(W['h']+b['h'])
print(vectors)
