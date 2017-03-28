'''
"Building powerful image classification models using very little data"
from blog.keras.io.
It uses data that can be downloaded at:
https://www.kaggle.com/c/dogs-vs-cats/data
In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created cats/ and dogs/ subfolders inside train/ and validation/
In summary, this is our directory structure:
```
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
```
'''
#encoding=utf-8
import numpy as np
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.utils import plot_model
import sys
from  vgg16 import VGG16
import argparse

"""
Configuration settings
"""
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-dp','--data_path',action='store',type=str,
        default='../../data/keras',help='train and val  file path')
parser.add_argument('-lr','--learning_rate',action='store',type=float,
        default=0.0001,help='learning_rate')
parser.add_argument('-mt','--momentum',action='store',type=float,
        default=0.9,help='learning_rate')
parser.add_argument('-ne','--num_epochs',action='store',type=int,
        default=50,help='num_epochs')
parser.add_argument('-bs','--batch_size',action='store',type=int,
        default=128,help='batch size')
parser.add_argument('-nc','--num_classes',action='store',type=int,
        default=2,help='num classes')   # no use now
parser.add_argument('-tl','--train_layers',nargs='+',action='store',type=str,
        default=['fc1','fc2','logit'],help='layers need to be trained.')
parser.add_argument('-tn','--top_N',action='store',type=int,
        default=5,help='whether the targets are in the top K predictions.')
parser.add_argument('-rc','--restore_checkpoint',action='store',type=str,
        default='',help='use restore mode to initialize weights.\nex: python finetune.py -rc ../../data/checkpoint/model_epoch1.ckpt')

args = parser.parse_args()
print("="*50)
print("[INFO] args:\r")
print(args)
print("="*50)

train_data_dir = args.data_path + '/train'
validation_data_dir = args.data_path + '/test'

epochs = args.num_epochs
batch_size = args.batch_size

train_layers = args.train_layers

learning_rate  = args.learning_rate
momentum = args.momentum


def preprocess_input_vgg(x):
    """Wrapper around keras.applications.vgg16.preprocess_input()
    to make it compatible for use with keras.preprocessing.image.ImageDataGenerator's
    `preprocessing_function` argument.
    
    Parameters
    ----------
    x : a numpy 3darray (a single image to be preprocessed)
    
    Note we cannot pass keras.applications.vgg16.preprocess_input()
    directly to to keras.preprocessing.image.ImageDataGenerator's
    `preprocessing_function` argument because the former expects a
    4D tensor whereas the latter expects a 3D tensor. Hence the
    existence of this wrapper.
    
    Returns a numpy 3darray (the preprocessed image).
    
    """
    from keras.applications.imagenet_utils import preprocess_input
    X = np.expand_dims(x, axis=0) # 3D 2 4D
    X = preprocess_input(X) # only 4D tensor ,RGB2BGR ,center:103 116 123 
    return X[0]


# vgg16 
vgg16 = VGG16(weights='imagenet')

# ** get vgg top layer then add a logit layer for classification **
fc2 = vgg16.get_layer('fc2').output
prediction = Dense(output_dim=1, activation='sigmoid', name='logit')(fc2)
model = Model(input=vgg16.input, output=prediction)

# which layer will be trained 
for layer in model.layers:
    #if layer.name in ['fc1', 'fc2', 'logit']:
    if layer.name in train_layers:
        continue
    layer.trainable = False

# model summary and structure pic.
model.summary()
# only can be used in py2
#plot_model(model, show_shapes=True, show_layer_names=True,to_file='model.png')

# compile
sgd = SGD(lr=learning_rate, momentum=0.9)
model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

# data generation
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input_vgg,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(directory=train_data_dir,
                                                    target_size=[224, 224],
                                                    batch_size=batch_size,
                                                    class_mode='binary')

validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input_vgg)

validation_generator = validation_datagen.flow_from_directory(directory=validation_data_dir,
                                                              target_size=[224, 224],
                                                              batch_size=batch_size,
                                                              class_mode='binary')

# begin to fit 
model.fit_generator(train_generator,
                    steps_per_epoch=200,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=16);
