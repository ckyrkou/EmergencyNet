import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.models import load_model,save_model
from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,LearningRateScheduler,TerminateOnNaN,ReduceLROnPlateau, TensorBoard

from tensorflow.keras import backend as K

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.resnet import ResNet50

from tensorflow.keras.callbacks import LearningRateScheduler,ModelCheckpoint
from tensorflow.keras.losses import CategoricalCrossentropy

import cv2

import random as rnd
import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn.metrics import classification_report, confusion_matrix

#https://github.com/mjkvaak/ImageDataAugmentor
from ImageDataAugmentor.image_data_augmentor import *

from augment import create_augmentations

from model import make_ACFF_model
from other import *
K.clear_session()

train_data_dir='../data/AIDER/'
val_data_dir='../data/AIDER/'

img_height=240
img_width=240
num_classes = 5
num_workers=1
batch_size=64
epochs = 200
lr_init=1e-2

inp,cls = make_ACFF_model(img_height,img_width,C=num_classes)
model = Model(inputs=[inp], outputs=[cls])
model.summary()

AUGMENTATIONS = create_augmentations(img_height,img_width,p=0.1)

seed = 22
rnd.seed(seed)
np.random.seed(seed)

dsplit = 0.2

train_datagen = ImageDataAugmentor(
        rescale=1./255.,
        augment=AUGMENTATIONS,
        validation_split=dsplit)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
    )

validation_datagen = ImageDataGenerator(rescale=1./255.,
    preprocessing_function = None,validation_split=dsplit)

validation_generator = validation_datagen.flow_from_directory(
    val_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
    )


checkpoint = ModelCheckpoint('../results/model.h5', monitor='val_categorical_accuracy', save_best_only=True, mode='max', verbose=1, save_weights_only=False)
weight_checkpoint = ModelCheckpoint('../results/model_weights.h5', monitor='val_categorical_accuracy', save_best_only=True, mode='max', verbose=1, save_weights_only=True)

opt = tf.keras.optimizers.SGD(lr=lr_init,momentum=0.9)
cd = cosine_decay(epochs_tot=epochs,initial_lrate=lr_init,period=1,fade_factor=1.,min_lr=1e-3)

lrs = LearningRateScheduler(cd,verbose=1)

lrr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=10, min_lr=1e-6,verbose=1)

callbacks_list = [lrs,checkpoint,weight_checkpoint]


SMOOTHING=0.1
loss = CategoricalCrossentropy(label_smoothing=SMOOTHING)

model.compile(optimizer=opt,metrics=keras.metrics.CategoricalAccuracy(),loss=loss)

history=model.fit(x=train_generator,\
                   steps_per_epoch=train_generator.samples // batch_size,epochs=epochs,\
                  verbose=1,validation_data=validation_generator,validation_steps = validation_generator.samples // batch_size,callbacks=callbacks_list,workers=num_workers,class_weight={
        0:1.,1:1.,2:1.,3:0.35,4:1.})

plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('../results/acc.png')
plt.clf()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('../results/loss.png')

model = load_model('../results/model.h5')

score = model.evaluate(validation_generator)
print(score)

Y_pred = model.predict(validation_generator, steps =validation_generator.samples,batch_size=1)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(validation_generator.classes, y_pred))
print('Classification Report')

target_names = ['collapsed_building','fire','flooded_areas','normal','traffic_incident',]

print(classification_report(validation_generator.classes, y_pred, target_names=target_names))
