import numpy as np
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, Activation, BatchNormalization, merge, add, Dropout
from keras.callbacks import TerminateOnNaN
from keras.models import Model
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import tensorflow as tf
mnist = tf.keras.datasets.mnist
(x_train, y_train_temp),(x_test, y_test_temp) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0
x_train, x_test = np.reshape(x_train, x_train.shape + (1,)), np.reshape(x_test, x_test.shape + (1,)) 

y_train, y_test =  np.zeros((y_train_temp.size, y_train_temp.max()+1)), np.zeros((y_test_temp.size, y_test_temp.max()+1))
y_train[np.arange(y_train_temp.size), y_train_temp] = 1
y_test[np.arange(y_test_temp.size), y_test_temp] = 1

digit_input = Input(shape=(28, 28, 1))
conv1 = Conv2D(6, kernel_size=(3,3), strides=1)(digit_input)
relu1 = Activation('relu')(conv1)
normalize1 = BatchNormalization(axis=2)(relu1)

conv2 = Conv2D(6, kernel_size=(3,3), strides=1, padding='same')(normalize1)
relu2 = Activation('relu')(conv2)
normalize2 = BatchNormalization(axis=2)(relu2)

conv3 = Conv2D(6, kernel_size=(3,3), strides=1, padding='same')(normalize2)
relu3 = Activation('relu')(conv3)
normalize3 = BatchNormalization(axis=2)(relu3)

res = add([normalize3, normalize1])

flat = Flatten()(res)
fc1 = Dense(128)(flat)
fc1 = Dropout(0.3)(fc1)
relu1 = Activation('relu')(fc1)
output = Dense(10, activation='softmax')(relu1)

model = Model(inputs=digit_input, outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())


model_json = model.to_json()
with open("models/mnist6.json", "w") as json_file:
    json_file.write(model_json)


history = model.fit(x_train, y_train, epochs=24, batch_size=128,callbacks=[TerminateOnNaN()])
score, acc = model.evaluate(x_test, y_test)

print ("accuracy" , acc)
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
plt.matshow(cm)
# accuracy 0.988
# [[ 974    1    2    0    0    0    1    1    1    0]
#  [   0 1132    0    1    0    0    1    0    1    0]
#  [   0    1 1025    0    0    0    0    3    3    0]
#  [   0    0    3 1001    0    3    0    0    3    0]
#  [   1    0    3    0  977    0    0    0    1    0]
#  [   1    0    1    5    0  881    1    0    2    1]
#  [   6    3    0    0    1    4  938    0    6    0]
#  [   3    4   11    1    1    0    0  999    2    7]
#  [   1    0    3    0    0    0    0    2  967    1]
#  [   4    4    0    1    6    4    0    1    4  985]]