import h5py
import numpy as np
from i3d_inception import Inception_Inflated3d
import keras
from keras import optimizers
from keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger
from keras.preprocessing.image import ImageDataGenerator

hf = h5py.File("dummy.h5","r")
train_label_file = open("train_label.txt","r")
test_label_file = open("test_label.txt","r")
validation_label_file = open("validation_label.txt","r")
train_raw_labels = train_label_file.read().split("\n")
test_raw_labels = test_label_file.read().split("\n")
validation_raw_labels = validation_label_file.read().split("\n")
train_labels = keras.utils.to_categorical(np.array(train_raw_labels),num_classes = 10)
test_labels = keras.utils.to_categorical(np.array(test_raw_labels),num_classes = 10)
validation_labels = keras.utils.to_categorical(np.array(validation_raw_labels),num_classes = 10)

FRAME_HEIGHT = 224
FRAME_WIDTH = 224
NUM_FRAMES = 126
NUM_CLASSES = 10
BATCH = 4


def generator(type):
  i=0
  counter = 0
  if type=="train":
    while True:
      batch_features = np.zeros((BATCH,NUM_FRAMES, FRAME_WIDTH, FRAME_HEIGHT,3))
      batch_labels = np.zeros((BATCH,NUM_CLASSES))
      for i in range(BATCH):
        batch_features[i]= hf["train"][counter%30]
        batch_labels[i] = train_labels[counter%30]
        print("Index: "+str(i)+", Counter: "+str(counter))
        print(batch_labels)
        counter+=1
      yield batch_features,batch_labels
  elif type=="test":
    while True:
      batch_features = np.zeros((1,NUM_FRAMES, FRAME_WIDTH, FRAME_HEIGHT,3))
      batch_labels = np.zeros((1,NUM_CLASSES))
      for i in range(1):
        batch_features[i]= hf["test"][counter%10]
        batch_labels[i] = test_labels[counter%10]
        print("Index: "+str(i))
        print(batch_labels)
        counter+=1
      yield batch_features,batch_labels
  elif type=="validation":
    while True:
      batch_features = np.zeros((1,NUM_FRAMES, FRAME_WIDTH, FRAME_HEIGHT,3))
      batch_labels = np.zeros((1,NUM_CLASSES))
      for i in range(1):
        batch_features[i]= hf["validation"][counter%10]
        batch_labels[i] = validation_labels[counter%10]
        print("Index: "+str(i))
        print(batch_labels)
        counter+=1
      yield batch_features,batch_labels

rgb_model = Inception_Inflated3d(
            include_top=False,
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

rgb_model.fit_generator(generator("train"), steps_per_epoch=len(hf["train"])//4, epochs=10, callbacks=callbacks_list,shuffle=True,validation_data = generator("validation"),validation_steps=1)

score = rgb_model.predict_generator(generator("test"),steps=10)
np.save("dummy_result",score)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])

hf.close()
