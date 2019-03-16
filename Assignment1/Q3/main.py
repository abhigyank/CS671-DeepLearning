# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST/", one_hot=True)
import tensorflow as tf
from api import dense
import numpy as np
import matplotlib.pyplot as plt

alpha = 0.001
epochs = 500
batch_size = 128
display_step = 50

n_hidden_1 = 512 # 1st layer number of neurons
n_hidden_2 = 256 # 1st layer number of neurons
n_hidden_3 = 128 # 1st layer number of neurons
num_input = 784
num_classes = 10 
accuracies, losses = [],[]
itr = []
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

hidden1,_ = dense(X, num_input, n_hidden_1)
hidden2,_ = dense(hidden1, n_hidden_1, n_hidden_2)
hidden3,_ = dense(hidden2, n_hidden_2, n_hidden_3)
model,_ = dense(hidden3, n_hidden_3, num_classes)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=alpha)
train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
confusion_matix = tf.confusion_matrix(predictions=tf.argmax(model, 1), labels=tf.argmax(Y, 1))

init = tf.global_variables_initializer()

# Start training    

with tf.Session() as sess:

    sess.run(init)
    saver = tf.train.Saver()
    for step in range(1, epochs+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
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
    # Calculate accuracy for MNIST test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: mnist.test.images,
                                      Y: mnist.test.labels}))
    cm = sess.run(confusion_matix, feed_dict={X: mnist.test.images,
                                      Y: mnist.test.labels})
    saver.save(sess, 'models/mnist/mnist')
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