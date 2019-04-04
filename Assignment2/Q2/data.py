import numpy as np
import os
from PIL import Image
# import pandas as pd
from sklearn.model_selection import train_test_split
# from tensorflow.keras.utils import to_categorical
split = 0.75
dataset=[]
x=[]
y1=[]
y2=[]
y3=[]
y4=[]
path = './Data/'
for class_ in os.listdir(path):
	cnt=1
	print (class_)
	for files in os.listdir(path + class_):
		# print (cnt)
		if cnt==500:
			break
		data_pt=[]
		img = Image.open(path + class_ + '/' + files)
		aa = np.asarray(img)
		# data_pt.append(aa)
		x.append(aa)
		temp=files.split("_")
		del temp[-1]
		for i in range(len(temp)):
			temp[i]=int(temp[i])
		# print temp
		one_hot=[]
		one_hot.append(np.asarray([0,0]))
		one_hot.append(np.asarray([0,0]))
		one_hot.append(np.asarray([0]*12))
		one_hot.append(np.asarray([0,0]))
		for i in range(len(temp)):
			one_hot[i][temp[i]]=1
		y1.append(one_hot[0])
		y2.append(one_hot[1])
		y3.append(one_hot[2])
		y4.append(one_hot[3])
		# data_pt.append(np.asarray(one_hot))
		# dataset.append(np.asarray(data_pt))
		img.close()
		cnt+=1
# print (np.array(dataset)[:,0].shape)
# k = np.array([])
# for i in np.array(dataset)[:,0]:
# 	np.append(k, i, axis=0)
# print (k.shape)
# exit()
np.save('TRAIN.npy',x)
np.save('y1.npy',y1)
np.save('y2.npy',y2)
np.save('y3.npy',y3)
np.save('y4.npy',y4)
exit()

# train,test=dataset[int(len(dataset) * .75) : int(len(dataset) * .25)]
train=np.asarray(dataset[:int(len(dataset) * .75)])
test=np.asarray(dataset[(int(len(dataset) * .75)):])
from random import shuffle
shuffle(train)
shuffle(test)
TRAIN=[]
# TRAIN[0]=[]
# TRAIN["length"]=[]
# TRAIN["color"]=[]
# TRAIN["angle"]=[]
# TRAIN["width"]=[]
TRAIN.append([])
TRAIN.append([])
TRAIN.append([])
TRAIN.append([])
TRAIN.append([])
# TEST={}
# TEST["X"]=[]
# TEST["length"]=[]
# TEST["color"]=[]
# TEST["angle"]=[]
# TEST["width"]=[]
TEST=[]
TEST.append([])
TEST.append([])
TEST.append([])
TEST.append([])
TEST.append([])
# print train
exit()
for i in train:
	# print i
	TRAIN[0].append(np.asarray(i[0]))
	TRAIN[1].append(np.asarray(i[1][0]))
	TRAIN[2].append(np.asarray(i[1][1]))
	TRAIN[3].append(np.asarray(i[1][2]))
	TRAIN[4].append(np.asarray(i[1][3]))
	# exit()
for i in test:
	# print i
	TEST[0].append(np.asarray(i[0]))
	TEST[1].append(np.asarray(i[1][0]))
	TEST[2].append(np.asarray(i[1][1]))
	TEST[3].append(np.asarray(i[1][2]))
	TEST[4].append(np.asarray(i[1][3]))


print (np.asarray(TRAIN)[0][0].shape)
exit()
np.save('TRAIN.npy', np.asarray(TRAIN))
np.save('TEST.npy', np.asarray(TEST))
# print TRAIN
