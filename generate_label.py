import cv2
import os
import h5py
from os import walk, listdir, mkdir
from os.path import isfile, join, isdir
from sklearn import preprocessing
import numpy as np

directories = [f for f in listdir(os.getcwd() + '/dataset') if os.path.isdir(os.getcwd() + join('/dataset', f))]
count = 0

f = open("label.txt","w")

for dir in directories:
	directory = '/dataset/'+dir
	onlyfiles = [f for f in listdir(os.getcwd() + directory) if isfile(os.getcwd() + join(directory, f))]
	for file in onlyfiles:
		f.write(str(count)+"\n")
	count+=1
