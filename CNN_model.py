
from keras.models import Model
from keras.layers import Input,Dense,Flatten, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Activation,Conv2DTranspose, BatchNormalization,Dropout, Lambda
from tensorflow.keras.optimizers import Adam

def CNN(input_size):
    inputs=Input(input_size)
    
    C1=Conv2D(16, (3,3), activation='relu', padding='same')(inputs)
    C1=MaxPooling2D()(C1)
    
    C2=Conv2D(32, (3,3), activation='relu', padding='same')(C1)
    C2=MaxPooling2D()(C2)
    
    C3=Conv2D(64, (3,3), activation='relu', padding='same')(C2)
    C3=MaxPooling2D()(C3)
    
    F1=Flatten()(C3)
    FC1=Dense(1,activation='sigmoid')(F1)
    
    model=Model(inputs,FC1,name='CNN')
    return model

