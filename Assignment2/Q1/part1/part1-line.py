import numpy as np
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, Activation, BatchNormalization
from keras.models import Model
from tensorflow.examples.tutorials.mnist import input_data
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
linePath = "../LineData/"

lineData =  {
    "train": np.load(linePath + 'train.npy'),
    "test": np.load(linePath + 'test.npy'),
    "Y_test": np.load(linePath + 'Y_test.npy'),
    "Y_train": np.load(linePath +'Y_train.npy'),
}

x_train, y_train, x_test, y_test = lineData["train"], lineData["Y_train"], lineData["test"], lineData["Y_test"], 

digit_input = Input(shape=(28, 28, 3))
conv1 = Conv2D(32, kernel_size=(7,7), strides=1)(digit_input)
relu1 = Activation('relu')(conv1)
normalize1 = BatchNormalization(axis=2)(relu1)
pool1 = MaxPooling2D(pool_size=(2,2), strides=2)(normalize1)
flat = Flatten()(pool1)
fc1 = Dense(1024)(flat)
relu1 = Activation('relu')(fc1)
output = Dense(96, activation='softmax')(relu1)

model = Model(inputs=digit_input, outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

history = model.fit(x_train, y_train, epochs=7, batch_size=128, shuffle=True)

score, acc = model.evaluate(x_test, y_test, batch_size=128)
model.save('model/line.h5')
print ("accuracy" , acc)
# accuracy 0.990625

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