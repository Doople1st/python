import h5py
import numpy as np
from i3d_inception import Inception_Inflated3d
import keras

hf = h5py.File("data.h5","r")
label_file = open("label.txt","r")
raw_labels = label_file.read().split("\n")
labels = np.array(raw_labels)
FRAME_HEIGHT = 224
FRAME_WIDTH = 224
NUM_FRAMES = 242
NUM_CLASSES = len(labels)
BATCH = 20
#Store data index for testing
temp1 = []
#Store data index for training
temp2 = []
for i in range(2560):
	mod = i % 40
	if mod>=32 and mod<=39:
		temp1.append(i)
	else:
		temp2.append(i)

train_index = np.array(temp2)
test_index = np.array(temp1)


def train_generator(features, labels, batch_size):
 batch_features = np.zeros((batch_size, NUM_FRAMES, FRAME_WIDTH, FRAME_HEIGHT,3))
 batch_labels = np.zeros((batch_size,1))
 while True:
   for i in range(batch_size):
     # choose random index in features
     rand = np.random.choice(len(train_index),1)
     index = int(train_index[rand])
     #print(index)
     batch_features[i] = features[index]
     #print(labels[index])
     batch_labels[i] = int(labels[index])
   yield batch_features, batch_labels

def eval_generator(features, labels, batch_size):
 batch_features = np.zeros((batch_size, NUM_FRAMES, FRAME_WIDTH, FRAME_HEIGHT,3))
 batch_labels = np.zeros((batch_size,1))
 while True:
   for i in range(batch_size):
     # choose random index in features
     rand = np.random.choice(len(test_index),1)
     index = int(test_index[rand])
     batch_features[i] = features[index]
     batch_labels[i] = int(labels[index])
   yield batch_features, batch_labels

rgb_model = Inception_Inflated3d(
            include_top=False,
            weights='rgb_kinetics_only',
            input_shape=(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, 3),
            classes=NUM_CLASSES)

rgb_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

rgb_model.fit_generator(train_generator(hf["videos"], labels, BATCH), steps_per_epoch=BATCH, epochs=10)

score = rgb_model.evaluate_generator(eval_generator(hf["videos"],labels,BATCH),steps=10)
print('Test loss:', score[0])
print('Test accuracy:', score[1])