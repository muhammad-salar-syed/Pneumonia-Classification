
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input

img_size = 224
train_gen = ImageDataGenerator(rescale = 1/255.,brightness_range=[0.5,1.5], rotation_range=25,horizontal_flip=True,
    vertical_flip=True,width_shift_range=0.1, height_shift_range=0.1)


test_gen = ImageDataGenerator(rescale = 1/255.)

# Generator
train_data_generator = train_gen.flow_from_directory('./chest_xray/train',
                                                     target_size = (img_size, img_size),
                                                     batch_size = 4,
                                                     shuffle=True,
                                                     color_mode ="rgb",
                                                     class_mode='binary',
                                                     seed=24)


test_data_generator = test_gen.flow_from_directory('./chest_xray/test',
                                                   target_size = (img_size, img_size),
                                                   batch_size = 4,
                                                   shuffle=False,
                                                   color_mode ="rgb",
                                                   class_mode='binary',
                                                   seed=24)
# Defining labels
labels = ['Normal', 'Pneumonia']
samples = train_data_generator.__next__()
images = samples[0]
target = samples[1]

plt.figure(figsize=(12,12))
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    plt.imshow(images[i])
    plt.title('Class: {}'.format(labels[int(target[i])]),fontsize = 12)
    plt.axis('off')
    
#plt.savefig('./fig.png')

# fit model
from VGG16_model import TF_VGG16,TF_VGG16_finetune
from CNN_model import CNN
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

early_stop = EarlyStopping(monitor='val_loss', patience=4, verbose=1)
log_csv = CSVLogger('./TF_VGG16_finetune_logs.csv', separator=',', append=False)
callbacks_list = [early_stop, log_csv]

input_shape=images[0,:,:,:].shape
print(input_shape)

model=TF_VGG16_finetune(input_shape)
model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics='accuracy')

history = model.fit(train_data_generator, validation_data=test_data_generator, epochs=20,callbacks=callbacks_list)
model.save('./TF_VGG16_finetune.hdf5')

