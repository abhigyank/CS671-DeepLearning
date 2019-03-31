import numpy as np
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, Activation, BatchNormalization, Dropout, add
from keras.callbacks import TerminateOnNaN
from keras.models import Model
from keras.optimizers import SGD, Adam
import matplotlib.pyplot as plt
import keras.backend as K
from sklearn.metrics import confusion_matrix
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
adam = Adam(lr=0.001, decay=0, epsilon=1e-4)

model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

print(model.summary())
ac = []
los = []

for i in range(20):
    hist = model.fit(x_train, y_train, epochs=1, batch_size=128,callbacks=[TerminateOnNaN()])
    ac.append(hist.history['acc'])
    los.append(hist.history['loss'])
    if(i >= 3 and i%3==0):
        K.set_value(adam.lr, 0.5 * K.get_value(adam.lr))
    # if(i == 0):
    #     K.set_value(adam.clipnorm, 0.3 * K.get_value(adam.clipnorm))

score, acc = model.evaluate(x_test, y_test)

model.save('models/line3.h5')

print ("accuracy" , acc)

plt.plot(ac)
plt.title('Training Accuracy')
plt.show()
plt.plot(los)
plt.title('Training Loss')
plt.show()

y_pred = model.predict(x_test)
cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
print (cm)