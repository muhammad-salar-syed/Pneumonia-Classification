
from keras.models import Model
from keras.layers import Input,Dense,Flatten, Conv2D,GlobalAveragePooling2D, MaxPooling2D, UpSampling2D, Concatenate, Activation,Conv2DTranspose, BatchNormalization,Dropout, Lambda
from tensorflow.keras.optimizers import Adam
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19

def TF_VGG16(input_shape):

    conv_base = VGG16(include_top=False,
                     weights='imagenet', 
                     input_shape=input_shape)
    
    conv_base.trainable=False
    conv_base_output = conv_base.output
    F = Flatten()(conv_base_output)
    FC1 = Dense(50, activation='relu')(F)
    FC2 = Dense(20, activation='relu')(FC1)
    output_layer = Dense(1, activation='sigmoid')(FC2)

    model = Model(inputs=conv_base.input, outputs=output_layer)
    
    return model


def TF_VGG16_finetune(input_shape):

    conv_base = VGG16(include_top=False,
                     weights='imagenet', 
                     input_shape=input_shape)
    for layer in conv_base.layers[:15]:
        layer.trainable = False
        
    conv_base_output = conv_base.output
    F = Flatten()(conv_base_output)
    FC1 = Dense(50, activation='relu')(F)
    FC2 = Dense(20, activation='relu')(FC1)
    output_layer = Dense(1, activation='sigmoid')(FC2)

    model = Model(inputs=conv_base.input, outputs=output_layer)
    
    return model