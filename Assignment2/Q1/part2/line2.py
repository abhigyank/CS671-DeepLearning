import numpy as np
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, Activation, BatchNormalization, Dropout, add
from keras.callbacks import TerminateOnNaN
from keras.models import Model
from keras.optimizers import SGD, Adam
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
linePath = "../LineData/"

lineData =  {
    "train": np.load(linePath + 'train.npy'),
    "test": np.load(linePath + 'test.npy'),
    "Y_test": np.load(linePath + 'Y_test.npy'),
    "Y_train": np.load(linePath + 'Y_train.npy'),
}

x_train, y_train, x_test, y_test = lineData["train"], lineData["Y_train"], lineData["test"], lineData["Y_test"], 

digit_input = Input(shape=(28, 28, 3))
conv1 = Conv2D(8, kernel_size=(3,3), strides=1)(digit_input)
relu1 = Activation('relu')(conv1)
normalize1 = BatchNormalization(axis=2)(relu1)
pool1 = MaxPooling2D(pool_size=(2,2), strides=2)(normalize1)

conv2 = Conv2D(8, kernel_size=(3,3), strides=1, padding='same')(pool1)
relu2 = Activation('relu')(conv2)

conv2 = Conv2D(8, kernel_size=(3,3), strides=1, padding='same')(relu2)
relu2 = Activation('relu')(conv2)

res = add([relu2,pool1])

conv2 = Conv2D(16, kernel_size=(3,3), strides=1, padding='same')(res)
relu2 = Activation('relu')(conv2)

conv3 = Conv2D(16, kernel_size=(3,3), strides=1, padding='same')(relu2)
relu3 = Activation('relu')(conv3)

conv3 = Conv2D(16, kernel_size=(3,3), strides=1, padding='same')(relu3)
relu3 = Activation('relu')(conv3)

res2 = add([relu3,relu2])

flat = Flatten()(res2)
fc1 = Dense(128, activation='relu')(flat)
fc1 = Dropout(0.3)(fc1)

output = Dense(96, activation='softmax')(fc1)

model = Model(inputs=digit_input, outputs=output)
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0005), metrics=['accuracy'])

print(model.summary())

history = model.fit(x_train, y_train, epochs=12, batch_size=128,callbacks=[TerminateOnNaN()])
model.save('models/line2.h5')

score, acc = model.evaluate(x_test, y_test)

print( "accuracy" , acc)
# accuracy 0.9976388888888889

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
# [[150   0   0 ...   0   0   0]
#  [  0 150   0 ...   0   0   0]
#  [  0   0 150 ...   0   0   0]
#  ...
#  [  0   0   0 ... 150   0   0]
#  [  0   0   0 ...   0 150   0]
#  [  0   0   0 ...   0   0 150]]