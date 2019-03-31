import numpy as np
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, Activation, BatchNormalization
from keras.models import Model
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pyplot as plt
mnist = tf.keras.datasets.mnist
(x_train, y_train_temp),(x_test, y_test_temp) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0
x_train, x_test = np.reshape(x_train, x_train.shape + (1,)), np.reshape(x_test, x_test.shape + (1,)) 

y_train, y_test =  np.zeros((y_train_temp.size, y_train_temp.max()+1)), np.zeros((y_test_temp.size, y_test_temp.max()+1))
y_train[np.arange(y_train_temp.size), y_train_temp] = 1
y_test[np.arange(y_test_temp.size), y_test_temp] = 1

digit_input = Input(shape=(28, 28, 1))
conv1 = Conv2D(32, kernel_size=(3,3), strides=1)(digit_input)
relu1 = Activation('relu')(conv1)
conv2 = Conv2D(64, kernel_size=(3,3), strides=1)(relu1)
relu2 = Activation('relu')(conv2)
normalize1 = BatchNormalization(axis=2)(relu2)
pool1 = MaxPooling2D(pool_size=(2,2), strides=2)(normalize1)
flat = Flatten()(pool1)
fc1 = Dense(1024)(flat)
relu3 = Activation('relu')(fc1)
output = Dense(10, activation='sigmoid')(relu3)

model = Model(inputs=digit_input, outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())
model_json = model.to_json()
with open("models/mnist1.json", "w") as json_file:
    json_file.write(model_json)

history = model.fit(x_train, y_train, epochs=8, batch_size=128)
score, acc = model.evaluate(x_test, y_test, batch_size=128)

print ("accuracy" , acc)
# accuracy 0.9878
from sklearn.metrics import confusion_matrix

plt.plot(history.history['acc'])
plt.title('Training Accuracy')
plt.show()
plt.plot(history.history['loss'])
plt.title('Training Loss')
plt.show()

y_pred = model.predict(x_test)
cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
print (cm)
# [[ 978    0    1    0    0    0    0    1    0    0]
#  [   0 1130    0    0    0    1    2    1    1    0]
#  [   0    3 1022    0    0    0    0    3    4    0]
#  [   0    1    1  999    0    6    0    0    3    0]
#  [   0    0    0    0  974    0    2    0    2    4]
#  [   1    0    0    4    0  885    2    0    0    0]
#  [   8    1    0    0    1    3  941    0    4    0]
#  [   0    6   11    0    0    0    0 1006    5    0]
#  [   4    0    1    0    0    0    0    1  967    1]
#  [   1    0    0    0    3   11    0    5   13  976]]