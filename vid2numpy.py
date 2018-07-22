import cv2
import os
import h5py
from os import walk, listdir, mkdir
from os.path import isfile, join, isdir
from sklearn import preprocessing
import numpy as np

directories = [f for f in listdir(os.getcwd() + '/dataset') if os.path.isdir(os.getcwd() + join('/dataset', f))]

a = np.load('v_CricketShot_g04_c01_rgb.npy')
print (a.shape);

min=0
max=242
count = 0
data=[]
data_shape= (2560,242,224,224,3)
data_file = []
frames = 0
first = True

hf = h5py.File('data.h5', 'w')
hf.create_dataset('videos',data_shape,np.float16)

def normalize(x):
	return (x.astype(float)-128.0)/255.0

for dir in directories:
	directory = '/dataset/'+dir
	onlyfiles = [f for f in listdir(os.getcwd() + directory) if isfile(os.getcwd() + join(directory, f))]
	# print(directory)
	numpy_array = np.array(data)
	print(numpy_array.shape)
	for file in onlyfiles:
		filepath = directory+'/'+file;
		# print(filepath + ", Max: "+str(max)+", Min: "+str(min))
		print(filepath)
		# print(filepath)
		vidcap = cv2.VideoCapture(os.getcwd() + filepath)
		success = True
		data_file = []
		while success:
			vidcap.grab()
			success, image = vidcap.retrieve()  # success, image = vidcap.read()
			if success:
				image = cv2.resize(image,(224,224))
				image = image.astype(float)
				for i in range(len(image)):
					for j in range(len(image[i])):
						for k in range(len(image[i][j])):
							image[i][j][k] = normalize(image[i][j][k])
				data_file.append(image)
				frames+=1
			else:
				if frames < max:
					temp = max - frames
					blank = np.zeros((224,224,3))
					for i in range(temp):
						data_file.append(blank)
				frames=0
				hf["videos"][count]=data_file
				count+=1
hf.close()

