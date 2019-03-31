import numpy as np
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, Activation, BatchNormalization
from keras.models import Model
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

mnist = tf.keras.datasets.mnist
(x_train, y_train_temp),(x_test, y_test_temp) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0
x_train, x_test = np.reshape(x_train, x_train.shape + (1,)), np.reshape(x_test, x_test.shape + (1,)) 

y_train, y_test =  np.zeros((y_train_temp.size, y_train_temp.max()+1)), np.zeros((y_test_temp.size, y_test_temp.max()+1))
y_train[np.arange(y_train_temp.size), y_train_temp] = 1
y_test[np.arange(y_test_temp.size), y_test_temp] = 1

digit_input = Input(shape=(28, 28, 1))
conv1 = Conv2D(32, kernel_size=(7,7), strides=1)(digit_input)
relu1 = Activation('relu')(conv1)
normalize1 = BatchNormalization(axis=2)(relu1)
pool1 = MaxPooling2D(pool_size=(2,2), strides=2)(normalize1)
flat = Flatten()(pool1)
fc1 = Dense(1024)(flat)
relu1 = Activation('relu')(fc1)
output = Dense(10, activation='softmax')(relu1)

model = Model(inputs=digit_input, outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

history = model.fit(x_train, y_train, epochs=8, batch_size=128)
score, acc = model.evaluate(x_test, y_test, batch_size=128)
model.save('model/mnist.h5') 
print ("accuracy" , acc)


plt.plot(history.history['acc'])
plt.title('Training Accuracy')
plt.show()
plt.plot(history.history['loss'])
plt.title('Training Loss')
plt.show()

y_pred = model.predict(x_test)
cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
print (cm)

TP = np.diag(cm) # True Positive
TP = np.asarray(TP, dtype="float32")
FP = np.sum(cm, axis=0) - TP # False Positive
FN = np.sum(cm, axis=1) - TP # False Negatives
precision = TP/(TP+FP)
recall = TP/(TP+FN)
print ("F_scores:", 2*precision*recall/(precision+recall))