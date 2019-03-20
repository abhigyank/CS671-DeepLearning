import numpy as np
import os
import glob
import cv2
path=[]
def to_image(arr):
	arr=np.asarray(arr)
	# img=arr.reshape(28*3,28*3,3)
	img=np.zeros((28*3,28*3,3),np.uint8)
	for i in range(3):
		for j in range(3):
			for k in range(28):
				for l in range(28):
					for m in range(3):
						img[k+28*i][l+28*j][m]=arr[i*3+j][k][l][m]
	return img
for i in range(1,97):
	path.append("Data/Class"+str(i)+"/")
res=[]
for c in path:
	count=0
	temp=[]
	for file in glob.glob(c+"*.jpg"):
		img=cv2.imread(file)
		temp.append(img)
		if len(temp)==9:
			res.append(to_image(temp))
			temp=[]
		count+=1
		if count==90:
			print "class_over"
			break
save_path="DataForVideo/"
counter=0
for i in res:
	print counter
	cv2.imwrite(os.path.join(save_path,"img"+str(counter)+".jpg"),i)
	counter+=1
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()