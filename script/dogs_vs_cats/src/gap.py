
# coding: utf-8

# In[1]:

from keras.models import *
from keras.layers import *
#from keras.applications import *
import keras.applications.inception_v3
from keras.applications.inception_v3 import InceptionV3,conv2d_bn
from keras.applications.inception_v3 import preprocess_input as inception_input

from keras.applications.resnet50 import ResNet50

from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input as xception_input

from keras.preprocessing.image import *

import h5py


# In[2]:
trainPath='/home/julyedu_51217/tensorlearn/script/vgg_finetune/data/train_v1'
testPath='/home/julyedu_51217/tensorlearn/script/vgg_finetune/data/testPic'

# for debug
#trainPath='/home/julyedu_51217/tensorlearn/script/vgg_finetune/data/testTiny'
#testPath='/home/julyedu_51217/tensorlearn/script/vgg_finetune/data/testTiny'

def write_gap(MODEL, image_size, lambda_func=None):

    input_shape = image_size + (3,) # (x,x) => (x,x,3)

    base_model = MODEL(input_shape=input_shape,weights='imagenet', include_top=False)
    model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output)) # model build, add a GAP layyer
    model.summary()

    print("[INFO] image generator.")
    gen = ImageDataGenerator(preprocessing_function=lambda_func)
    train_generator = gen.flow_from_directory(trainPath, image_size, shuffle=False, 
                                              batch_size=64) # class_model default is category  
    test_generator = gen.flow_from_directory(testPath, image_size, shuffle=False, 
                                             batch_size=64,
                                             class_mode=None) # no targets get yielded (only input images are yielded).
    print("[INFO] image generator finish.")

    print ("[INFO] begin to predict train data")
    train = model.predict_generator(train_generator,steps = train_generator.samples, verbose = 1)
    print(train.shape)

    print ("[INFO] begin to predict test data")
    test = model.predict_generator(test_generator, steps = test_generator.samples, verbose = 1)
    print(test.shape)

    with h5py.File("../data/gap_%s.h5"%MODEL.func_name,'w') as h:
        h.create_dataset("train", data=train)
        h.create_dataset("test", data=test)
        h.create_dataset("label", data=train_generator.classes)


# In[3]:

write_gap(ResNet50, (224, 224))

# In[4]:

write_gap(Xception, (299, 299), xception_input)


# In[5]:

write_gap(InceptionV3, (299, 299), inception_input)


# In[7]:

#write_gap(VGG16, (224, 224))


# In[8]:

#write_gap(VGG19, (224, 224))


# In[ ]:



