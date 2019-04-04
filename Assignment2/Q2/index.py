import numpy as np
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, Activation, BatchNormalization,Dropout
from keras.models import Model
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import backend as K
from sklearn.metrics import confusion_matrix
batch_size=100
# feature extractor
n_filters_conv1=32
n_filter_size_conv1=5
activation=tf.nn.relu
n_filters_conv2=64
n_filter_size_conv2=3

# color head
color_layer1_n_neurons=256
color_layer2_n_neurons=32

#wdth head 
wdth_layer1_n_neurons=256
wdth_layer2_n_neurons=32

# length head
length_layer1_n_neurons=256
length_layer2_n_neurons=32

# anle head
anle_layer1_n_neurons=256
anle_layer2_n_neurons=32
inp=np.load("TRAIN.npy")
y1,y2,y3,y4=np.load("y1.npy"),np.load("y2.npy"),np.load("y3.npy"),np.load("y4.npy")
x_train, x_test = inp[:int(0.8*len(inp))],inp[int(0.8*len(inp)):] 
y1_train, y1_test, y2_train, y2_test, y3_train, y3_test, y4_train, y4_test = y1[:int(0.8*len(y1))],y1[int(0.8*len(y1)):],y2[:int(0.8*len(y2))],y2[int(0.8*len(y2)):],y3[:int(0.8*len(y3))],y3[int(0.8*len(y3)):],y4[:int(0.8*len(y4))],y4[int(0.8*len(y4)):]  
# print(TRAIN[0])
# print(TRAIN[0])
# print TRAIN[0]
# train_input_fn = tf.estimator.inputs.numpy_input_fn(
# x={"x": np.array(TRAIN[0])},
# num_epochs=None,
# shuffle=True)

# TRAIN[0]=np.reshape(TRAIN[0],(-1,28,28,3))
# print(train_input_fn)
from PIL import Image
img_to_visualize=Image.open("Data/Class1/0_0_0_0_23.jpg")
img_to_visualize = np.asarray(img_to_visualize)



input_layer=Input(shape=(28, 28, 3))
conv1 = Conv2D(32, kernel_size=(5,5), strides=1,padding="same",activation="relu")(input_layer)
batch = BatchNormalization(axis=2)(conv1)
pool1 = MaxPooling2D(pool_size=(2,2), strides=2)(batch)
conv2 = Conv2D(64, kernel_size=(3,3), strides=1,activation="relu")(pool1)
pool2 = MaxPooling2D(pool_size=(2,2), strides=2)(conv2)
flat = Flatten()(pool2)

# color
dense_color_1=Dense(256,activation="relu")(flat)
dropout_color=Dropout(0.2)(dense_color_1)
dense_color_2=Dense(32,activation="relu")(dropout_color)
logits_color=Dense(2,activation="sigmoid",name="co")(dense_color_2)
# wdth_layer1_n_neurons	
dense_wdth_1=Dense(256,activation="relu")(flat)
dropout_wdth=Dropout(0.2)(dense_wdth_1)
dense_wdth_2=Dense(32,activation="relu")(dropout_wdth)
logits_wdth=Dense(2,activation="sigmoid",name="w")(dense_wdth_2)
# color
dense_length_1=Dense(256,activation="relu")(flat)
dropout_length=Dropout(0.2)(dense_length_1)
dense_length_2=Dense(32,activation="relu")(dropout_length)
logits_length=Dense(2,activation="sigmoid",name="le")(dense_length_2)
# color
dense_anle_1=Dense(256,activation="relu")(flat)
dropout_anle=Dropout(0.2)(dense_anle_1)
# dense_anle_2=Dense(32,activation="relu")(dropout_anle)
logits_anle=Dense(12,activation="softmax",name="an")(dense_anle_1)
    
model = Model(inputs=input_layer, outputs=[logits_color,logits_wdth,logits_length,logits_anle])
losses = {
	"co": "binary_crossentropy",
	"w": "binary_crossentropy",
	"le": "binary_crossentropy",
	"an": "categorical_crossentropy"
}
# lossWeights = {"co": 1.0, "w": 1.0, "le": 1.0, "co": 1.0}
model.compile(loss=losses, optimizer='adam', metrics=['accuracy'])
model.fit(x_train,{
	"co": y4_train,
	"w": y2_train,
	"le": y1_train,
	"an": y3_train
},epochs=1, shuffle=True, batch_size=128)
model.save('model.h5')
model.evaluate(x_test, {
	"co": y4_test,
	"w": y2_test,
	"le": y1_test,
	"an": y3_test
})
init = tf.global_variables_initializer()