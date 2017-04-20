# coding: utf-8

from keras.models import *
from keras.layers import *

import keras.applications.inception_v3
from keras.applications.inception_v3 import InceptionV3,conv2d_bn
from keras.applications.inception_v3 import preprocess_input as inception_input

from keras.applications.resnet50 import ResNet50

from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input as xception_input

from keras.preprocessing.image import *

import h5py
import yaml

import logging,datetime,sys,os,ConfigParser
from logging.handlers import TimedRotatingFileHandler

##############################
# logging
##############################
def myLog(logPath):

    logging.basicConfig(level=logging.DEBUG,
                format='[ %(asctime)s %(filename)s line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename=logPath,
                filemode='w')

    #定义一个StreamHandler，将INFO级别或更高的日志信息打印到标准错误，并将其添加到当前的日志处理对象#
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    return logging


##############################
# get model
##############################
class feature_generation(
        trainPath,
        testPath,
        version = 'base',
        genPath = None,
        gen_layer = 'notop',
        use_model = ['ResNet50','Xception','InceptionV3']
        ):

    self.gen_layer = gen_layer
    self.genPath =  genPath
    self.trainPath = trainPath
    self.testPath = testPath
    self.version = version
    self.use_model = use_model
    

    def get_model(MODEL,image_size):
        
        input_shape = image_size + (3,) # (x,x) => (x,x,3)
    
        base_model = MODEL(input_shape=input_shape,weights='imagenet', include_top=False)
        
        if gen_layer == 'notop':
            model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output)) # model build, add a GAP layyer
        elif gen_layer == 'fc':
            pass
        elif gen_layer == 'top':
            pass
    
        logger.info(model.summary())
        return model
    
    
    ##############################
    # genration
    ##############################
    def genration(lambda_func=None):
    
        logger.info("image generator.")
        gen = ImageDataGenerator(preprocessing_function=lambda_func)
        train_generator = gen.flow_from_directory(trainPath, image_size, shuffle=False, 
                                                  batch_size=1) # class_model default is category  
        test_generator = gen.flow_from_directory(testPath, image_size, shuffle=False, 
                                                 batch_size=1,
                                                 class_mode=None) # no targets get yielded (only input images are yielded).
        logger.info("image generator finish.")
    
        logger.info("begin to predict train data")
        train = model.predict_generator(train_generator,steps = train_generator.samples, verbose = 1)
        logger.info(train.shape)
    
        logger.info("begin to predict test data")
        test = model.predict_generator(test_generator, steps = test_generator.samples, verbose = 1)
        logger.info(test.shape)

         with h5py.File("../data/gap_{}_{}.h5".format(MODEL.func_name,version),'w') as h:
             h.create_dataset("train", data=train)
             h.create_dataset("test", data=test)
             h.create_dataset("label", data=train_generator.classes)




def write_gap(MODEL, image_size, lambda_func=None):
    '''
    old 
    '''

    input_shape = image_size + (3,) # (x,x) => (x,x,3)

    base_model = MODEL(input_shape=input_shape,weights='imagenet', include_top=False)
    model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output)) # model build, add a GAP layyer
    model.summary()

    logger.info("image generator.")
    gen = ImageDataGenerator(preprocessing_function=lambda_func)
    train_generator = gen.flow_from_directory(trainPath, image_size, shuffle=False, 
                                              batch_size=1) # class_model default is category  
    test_generator = gen.flow_from_directory(testPath, image_size, shuffle=False, 
                                             batch_size=1,
                                             class_mode=None) # no targets get yielded (only input images are yielded).
    logger.info("image generator finish.")

    logger.info("begin to predict train data")
    train = model.predict_generator(train_generator,steps = train_generator.samples, verbose = 1)
    logger.info(train.shape)

    logger.info("begin to predict test data")
    test = model.predict_generator(test_generator, steps = test_generator.samples, verbose = 1)
    logger.info(test.shape)

    with h5py.File("../data/gap_%s.h5"%MODEL.func_name,'w') as h:
        h.create_dataset("train", data=train)
        h.create_dataset("test", data=test)
        h.create_dataset("label", data=train_generator.classes)

    def process(self):

        for m in self.use_model:

            if m == "ResNet50":
                MODEL = ResNet50
                image_size = (224,224)
                model = get_model()
                genration(model,(224,224))

            elif m == "Xception":
                model = get_model(Xception, (299, 299))
                genration(model,(299,299))

            elif m == "InceptionV3":
                model = get_model(ResNet50, (299, 299))
                genration(model,(299,299))

if __name__ == "__main__":
    
    ##############################
    # CNF GET
    ##############################

    CNF = yaml.load(open('../conf/setting.yaml'))
    
    version = CNF['version']

    genPath = CNF['gen_path']
    trainPath = CNF['train_path']
    testPath = CNF['test_path']
    
    use_model = CNF['use_model']
    gen_layer = CNF['gen_layer']
    
    log_path = CNF['log_path']

    logger =  myLog(log_path+"/log_{}".format(version))  
    logger.info(CNF)

    for m in use_model:

        if m == "ResNet50":
            MODEL = ResNet50
            image_size = (224,224)
            model = get_model()
            genration(model,(224,224))

        elif m == "Xception":
            model = get_model(Xception, (299, 299))
            genration(model,(299,299))

        elif m == "InceptionV3":
            model = get_model(ResNet50, (299, 299))
           genration(model,(299,299))


#write_gap(Xception, (299, 299), xception_input)
#
#write_gap(InceptionV3, (299, 299), inception_input)

#write_gap(VGG16, (224, 224))

#write_gap(VGG19, (224, 224))

