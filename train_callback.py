import h5py
import numpy as np
from i3d_inception import Inception_Inflated3d
import keras
from keras import optimizers
from keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger

hf = h5py.File("data.h5","r")
label_file = open("label.txt","r")
raw_labels = label_file.read().split("\n")
numpy_labels = np.array(raw_labels)
labels = keras.utils.to_categorical(numpy_labels,num_classes = 64)
print(labels)
FRAME_HEIGHT = 224
FRAME_WIDTH = 224
NUM_FRAMES = 242
NUM_CLASSES = 64
BATCH = 64
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
  batch_labels = np.zeros((batch_size,64))
  count = 0
  while True:
    index = train_index[count%len(train_index)]
    yield features[index], labels[index]
    count+=1

def eval_generator(features, labels, batch_size):
  batch_features = np.zeros((batch_size, NUM_FRAMES, FRAME_WIDTH, FRAME_HEIGHT,3))
  batch_labels = np.zeros((batch_size,64))
  count = 0
  while True:
    index = test_index[count%len(test_index)]
    yield features[index], labels[index]

rgb_model = Inception_Inflated3d(
            include_top=True,
            weights='rgb_kinetics_only',
            input_shape=(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, 3),
            classes=NUM_CLASSES,endpoint_logit=False)

opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)

rgb_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=opt,
              metrics=['accuracy'])

best_checkpoint = ModelCheckpoint('weights_best.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
checkpoint = ModelCheckpoint('weights_epoch.hdf5', monitor='val_acc', verbose=1, save_best_only=False, mode='max')
csv_logger = CSVLogger('training.log', append=False)
tensorboard = TensorBoard(log_dir='./tf-logs')
callbacks_list = [checkpoint,best_checkpoint, csv_logger, tensorboard]

rgb_model.fit_generator(train_generator(hf["videos"], labels, 1), steps_per_epoch=2048//BATCH, epochs=100, callbacks=callbacks_list, verbose=1,shuffle=True)

score = rgb_model.evaluate_generator(eval_generator(hf["videos"],labels,1),steps=512//BATCH)
print('Test loss:', score[0])
print('Test accuracy:', score[1])