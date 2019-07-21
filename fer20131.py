#%%
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.layers.normalization import BatchNormalization

from keras.regularizers import l2 #, activity_l2
import keras
from keras.layers import Dense, Conv2D, UpSampling2D, MaxPooling2D, ZeroPadding2D, Flatten, Dropout
    
#%%
# Data Pre-processing
 
dataset_path = 'fer2013.csv'
image_size=(48,48)
 
def load_fer2013():
    data = pd.read_csv(dataset_path)
    pixels = data['pixels'].tolist()
    width, height = 48, 48
    faces = []
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.asarray(face).reshape(width, height)
        face = cv2.resize(face.astype('uint8'),image_size)
        faces.append(face.astype('float32'))
    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)
    emotions = pd.get_dummies(data['emotion']).as_matrix()
    return faces, emotions
 
def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x
 
#%%
  # Dependent Variables
#X_train = faces[0:28710,:]
#Y_train = emotions[0:28710]
#print(X_train.shape , Y_train.shape)
#%%
 # data split 
faces, emotions = load_fer2013()  
faces = preprocess_input(faces)
num_samples, num_classes=emotions.shape

xtrain, xtest,ytrain,ytest = train_test_split(faces, emotions,test_size=0.2,random_state=43,shuffle=True)

#%%
# Image data generator

data_generator = ImageDataGenerator(
                        featurewise_center=False,
                        featurewise_std_normalization=False,
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=.1,
                        horizontal_flip=True)

#%%
# Architecture

img_rows, img_cols = 48, 48

model=Sequential()

 
#1st convolution layer
model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48,48,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
 
#2nd convolution layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
 
#3rd convolution layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
 
model.add(Flatten())
 
#fully connected neural networks
model.add(Dense(1024))
model.add(Dropout(0.2))
model.add(Dense(1024))
model.add(Dropout(0.2))
 
model.add(Dense(7))
model.add(Activation('softmax'))
#%%

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

#%%

model.summary()
#%%
batch_size = 100
num_epochs = 300
input_shape = (48, 48, 1)
validation_split = .2
verbose = 1
num_classes = 7
patience = 50

#%%

data_generator.fit(xtrain)

model.fit_generator(data_generator.flow(xtrain, ytrain,
                                            batch_size),
                        steps_per_epoch=len(xtrain) / batch_size,
                        epochs=num_epochs, verbose=1,
                        validation_data=(xtest,ytest))
#%%

score = model.evaluate(xtest, ytest, verbose=0)

print('Test score:', score[0])
print('Test accuracy:', score[1])
#%%
model.save('fer2013model.h5')


#%%
from keras.utils import np_utils
accuracy = model['acc']
val_accuracy = model.model['val_acc']
loss = model.history['loss']
val_loss = model.['val_loss']
epochs = range(len(accuracy))
plt.plot( model, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
#%%
# summarize history for accuracy
from sklearn.metrics import classification_report,confusion_matrix
import itertools

Y_pred = model.predict(xtest)
print(Y_pred)
y_pred = np.argmax(Y_pred, axis=1)
print(y_pred)
#y_pred = model.predict_classes(X_test)
#print(y_pred)
target_names = ["Angry" ,"Disgust","Scared", "Happy", "Sad", "Surprised",
 "Neutral"]
					
print(classification_report(np.argmax(ytest,axis=1), y_pred,target_names=target_names))

print(confusion_matrix(np.argmax(ytest,axis=1), y_pred))

