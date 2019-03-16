import tensorflow as tf
from api import dense
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import random
import sys
split = 0.75

def shuffle():
    global train, Y_train
    randomize = np.arange(train.shape[0])
    np.random.shuffle(randomize)
    train = train[randomize]
    Y_train = Y_train[randomize]

def next_batch(X,Y,step):
    idx = ( step*batch_size ) % X.shape[0]
    if(idx == 0):
        shuffle()
        X = train
        Y = Y_train
    x,y = X[idx:idx+batch_size], Y[idx:idx+batch_size]
    return x,y

path = './Data1_numpy/'
# path = './data1_npy/'
train = np.load(path + 'train.npy')
test = np.load(path + 'test.npy')
Y_test = np.load(path + 'Y_test.npy')
Y_train = np.load(path + 'Y_train.npy')

alpha = 0.001
epochs = 2000 ## Reality - 10 epochs
batch_size = 128
display_step = 50

n_hidden_1 = 1024 
n_hidden_2 = 512 
n_hidden_3 = 256

num_input = 2352
num_classes = 96

accuracies, losses = [],[]

itr = []

X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])



hidden1,_ = dense(X, num_input, n_hidden_1, 0.0, activation=tf.nn.relu)
hidden2,_ = dense(hidden1, n_hidden_1, n_hidden_2, 0.0)
hidden3,_ = dense(hidden2, n_hidden_2, n_hidden_3, 0.0)
model,weights = dense(hidden3, n_hidden_3, num_classes, 0.0)

loss_op = tf.reduce_mean(tf.losses.softmax_cross_entropy(Y, model))
regularizer = tf.nn.l2_loss(weights)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha)
train_op = optimizer.minimize(loss_op + 0.1*regularizer)

correct_pred = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
confusion_matix = tf.confusion_matrix(predictions=tf.argmax(model, 1), labels=tf.argmax(Y, 1))

init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    saver = tf.train.Saver()    
    for step in range(1, epochs+1):
        batch_x, batch_y = next_batch(train, Y_train, step-1)

        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
            accuracies.append(acc)
            losses.append(loss)
            itr.append(step)
            print("Validation Accuracy:", \
                sess.run(accuracy, feed_dict={X: test,
                                      Y: Y_test}))

    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: test,
                                      Y: Y_test}))
    cm = sess.run(confusion_matix, feed_dict={X: test,
                                      Y: Y_test})

    for i in cm:
        if(sum(i)!=100): print "Invalid"
    saver.save(sess, 'models/q1/q1')    
    np.set_printoptions(threshold=sys.maxsize)
    print("Confussion Matrix:", cm)
    TP = np.diag(cm) # True Positive
    TP = np.asarray(TP, dtype="float32")
    FP = np.sum(cm, axis=0) - TP # False Positive
    FN = np.sum(cm, axis=1) - TP # False Negatives
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    print ("F_scores:", 2*precision*recall/(precision+recall))
    plt.plot(itr, accuracies)
    plt.title("Accuracies")
    plt.show()
    plt.plot(itr, losses)
    plt.title("Loss")
    plt.show()