import cv2
import numpy as np
from math import tan
from math import pi
import math
import random
import os
from PIL import Image
length=[7,15]
width=[1,3]
Angle=[15,30,45,60,75,90,105,120,135,150,165,180]
Angle=Angle[::-1] 
# print Angle
Color=[(0,0,255),(255,0,0)]
temp=1
for l in range(len(Color)):
	for k in range(len(Angle)):
		for j in range(len(width)):
			for i in range(len(length)):
				x2=int(length[i]*math.cos(Angle[k]*(math.pi/180)))
				y2=int(length[i]*math.sin(Angle[k]*(math.pi/180)))
				counter=0
				path="/home/lasers67/Desktop/Course/DL/Assignment1/Data/Class"+str(temp)+"/"
				os.mkdir(path)
				# print path
				while(counter<1000):
					x1=random.randint(0,28)
					y1=random.randint(0,28)
					x2=int(length[i]*math.cos(Angle[k]*(math.pi/180)))+x1
					y2=int(length[i]*math.sin(Angle[k]*(math.pi/180)))+y1
					if((x1<=(28-width[j]) and x2<=(28-width[j])) and (y1<=(28-width[j]) and y2<=(28-width[j])) and (x1>=width[j] and x2>=width[j] and y1>=width[j] and y2>=width[j])):
						img=np.zeros((28,28,3),np.uint8)
						cv2.line(img,(x1,y1),(x2,y2),Color[l],width[j])
						s=str(i)+"_"+str(j)+"_"+str(k)+"_"+str(l)+"_"+str(counter)+".jpg"
						counter+=1
						# im = Image.fromarray(img)
						# im.save(os.path.join(path,s),img)
						cv2.imwrite(os.path.join(path,s),img)
						# print os.path.join(path,s)
						# print s
						# cv2.imshow("img",img)
						cv2.waitKey(0)
						cv2.destroyAllWindows()
				temp+=1
# # count=0
# # for i in range(28):
# # 	for j in range(28):
# # 		if img[i][j][0]!=0:
# # 			print (i,j)
# # 			count=count+1
# # print (count)
	