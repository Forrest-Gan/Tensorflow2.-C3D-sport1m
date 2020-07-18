import  tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import layers, models, Input
from    tensorflow.keras.models import Model
from 	tensorflow.keras.layers import Conv3D, MaxPooling3D, Dense, Flatten, Dropout, ZeroPadding3D


def C3Dnet(nb_classes, input_shape):
    input_tensor = Input(shape=input_shape)
    # 1st block
    x = Conv3D(64, [3,3,3], activation='relu', padding='same', strides=(1,1,1), name='conv1')(input_tensor)
    x = MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2), padding='valid', name='pool1')(x)
    # 2nd block
    x = Conv3D(128, [3,3,3], activation='relu', padding='same', strides=(1,1,1), name='conv2')(x)
    x = MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='pool2')(x)
    # 3rd block
    x = Conv3D(256, [3,3,3], activation='relu', padding='same', strides=(1,1,1), name='conv3a')(x)
    x = Conv3D(256, [3,3,3], activation='relu', padding='same', strides=(1,1,1), name='conv3b')(x)
    x = MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='pool3')(x)
    # 4th block
    x = Conv3D(512, [3,3,3], activation='relu', padding='same', strides=(1,1,1), name='conv4a')(x)
    x = Conv3D(512, [3,3,3], activation='relu', padding='same', strides=(1,1,1), name='conv4b')(x)
    x= MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='pool4')(x)
    # 5th block
    x = Conv3D(512, [3,3,3], activation='relu', padding='same', strides=(1,1,1), name='conv5a')(x)
    x = Conv3D(512, [3,3,3], activation='relu', padding='same', strides=(1,1,1), name='conv5b')(x)
    x = ZeroPadding3D(padding=(0,1,1),name='zeropadding')(x)
    x= MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2), padding='valid', name='pool5')(x)
    # full connection
    x = Flatten()(x)
    x = Dense(4096, activation='relu',  name='fc6')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', name='fc7')(x)
    x = Dropout(0.5)(x)
    output_tensor = Dense(nb_classes, activation='softmax', name='fc8')(x)

    model = Model(input_tensor, output_tensor)
    return model

model=C3Dnet(487, (16, 112, 112, 3))
model.summary()
