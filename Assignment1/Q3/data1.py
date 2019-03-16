import tensorflow as tf
from api import dense
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
split = 0.75

train = []
test = []
Y_train = []
Y_test = []

path = './Data1/'

tot = 1

for class_ in os.listdir(path):
    gt = int(class_.split("Class")[1])
    cnt = 1
    print tot
    tot+=1
    for files in os.listdir(path + class_):
        img = Image.open(path + class_ + '/' + files)
        img = np.asarray(img)
        img = img.reshape(784*3)
        img = (img-np.ndarray.min(img))/(np.ndarray.max(img) - np.ndarray.min(img))
        # print img
        Y = [0]*96
        Y[gt-1] = 1
        Y = np.asarray(Y)
        if(cnt<=350):
            train.append(img)
            Y_train.append(Y)
        else:
            test.append(img)
            Y_test.append(Y)        
        cnt+=1
        if(cnt >500):break
    # break
# train, test, Y_test, Y_train = np.array(train), np.array(test), np.array(Y_test), np.array(Y_train)
# print train.shape 
np.save('data1_npy/train.npy', train)
np.save('data1_npy/test.npy', test)
np.save('data1_npy/Y_test.npy', Y_test)
np.save('data1_npy/Y_train.npy', Y_train)