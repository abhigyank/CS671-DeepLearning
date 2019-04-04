from keras.models import load_model
from keras import models
from keras import backend as K
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
model = load_model('model.h5')
img_path = 'Data/Class37/0_0_9_0_35.jpg'
img = image.load_img(img_path, target_size=(28, 28,3))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.
def get_activation(model,layer_index):
	layer_outputs = model.layers[layer_index].output 
	activation_model = models.Model(inputs=model.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input
	activations = activation_model.predict(img_tensor)
	num_channels=len(activations[0][0][0])
	leng=len(activations[0])
	arr=[]
	for i in range(num_channels):
		arr.append(np.zeros(shape=(leng,leng)))
	for i in range(leng):
		for j in range(leng):
			for k in range(num_channels):
				arr[k][i][j]=activations[0][i][j][k]*255.
	f, axarr =plt.subplots(int(num_channels/4),4)
	for i in range(int(num_channels/4)):
		for j in range(4):
			axarr[i,j].imshow(arr[i*(4)+j],interpolation="nearest",cmap='gray')
	plt.show()
get_activation(model,1)

def get_conv_filter(model,layer_index):
	MAX=-256
	MIN=256
	W=model.layers[layer_index].get_weights()[0]
	W = np.squeeze(W)
	filters=[]
	for i in range(len(W[0][0][0])):
		new_image=[]
		for j in range(len(W)):
			temp=[]
			for k in range(len(W)):
				temp.append(([0.0,0.0,0.0]))
			new_image.append(np.asarray(temp))
		new_image=np.asarray(new_image)
		filters.append(new_image)
	filters=np.asarray(filters)
	for i in range(len(W)):
		for j in range(len(W)):
			for k in range(len(W[0][0])):
				for l in range(len(W[0][0][0])):
					filters[l][i][j][k]=W[i][j][k][l]+0.5
	# print(MAX)
	# print(MIN)
	fig, axs = plt.subplots(8,4, figsize=(5,5))
	fig.subplots_adjust(hspace = .5, wspace=.001)
	axs = axs.ravel()
	for i in range(32):
	    axs[i].imshow(filters[i])
	    axs[i].set_title(str(i))
	plt.show()
get_conv_filter(model,1)
