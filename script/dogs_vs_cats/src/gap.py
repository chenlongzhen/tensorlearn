# coding: utf-8

from keras.models import *
from keras.layers import *

import keras.applications.inception_v3
from keras.applications.inception_v3 import InceptionV3,conv2d_bn
from keras.applications.inception_v3 import preprocess_input as inception_input

from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19

from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input as xception_input

from keras.preprocessing.image import *

import h5py
import yaml

import logging,datetime,sys,os,ConfigParser
from logging.handlers import TimedRotatingFileHandler
import timeit

start = timeit.default_timer()
print(start)

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
class feature_generation:

    def __init__(
        self,
        trainPath,
        testPath,
        version = 'base',
        genPath = 'None',
        gen_layer = 'notop',
        use_model = ['ResNet50','Xception','InceptionV3']
        ):

        self.gen_layer = gen_layer
        self.genPath =  genPath
        self.trainPath = trainPath
        self.testPath = testPath
        self.version = version
        self.use_model = use_model
    

    def get_model(self,MODEL,image_size):
        
        input_shape = image_size + (3,) # (x,x) => (x,x,3)
    

        if self.gen_layer == 'notop':
            base_model = MODEL(input_shape=input_shape,weights='imagenet', include_top=False)
            model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output)) # model build, add a GAP layyer

        elif self.gen_layer == 'fc':
            base_model = MODEL(input_shape=input_shape,weights='imagenet', include_top=True)
            fc = base_model.get_layer(index=-2) # get the last but one fc layer
            #base_model.pop() # future version, same with above
            model = Model(base_model.input, fc.output) 

        elif self.gen_layer == 'top':
            base_model = MODEL(input_shape=input_shape,weights='imagenet', include_top=True)
            model = base_model

        logger.info(model.summary())
        return model
    
    
    ##############################
    # genration
    ##############################
    def genration(self,MODEL,image_size,lambda_func=None):
        
        model = self.get_model(MODEL,image_size)
    
        logger.info("image generator.")
        if self.genPath == 'None':
            gen = ImageDataGenerator(preprocessing_function=lambda_func)
            train_generator = gen.flow_from_directory(self.trainPath, image_size, shuffle=False, 
                                                      batch_size=1) # class_model default is category  
            test_generator = gen.flow_from_directory(self.testPath, image_size, shuffle=False, 
                                                     batch_size=1,
                                                     class_mode=None) # no targets get yielded (only input images are yielded).
            logger.info("image generator finish.")
    
            logger.info("begin to predict train data")
            train = model.predict_generator(train_generator,steps = train_generator.samples, verbose = 1)
            logger.info("shape \r")
            logger.info(train.shape)
    
            logger.info("begin to predict test data")
            test = model.predict_generator(test_generator, steps = test_generator.samples, verbose = 1)
            logger.info("shape \r")
            logger.info(test.shape)

            with h5py.File("../data/model/gap_{}_{}.h5".format(MODEL.func_name,version),'w') as h:
                h.create_dataset("train", data=train)
                h.create_dataset("test", data=test)
                h.create_dataset("label", data=train_generator.classes)

        else:
            gen = ImageDataGenerator(preprocessing_function=lambda_func)
            test_generator = gen.flow_from_directory(self.testPath, image_size, shuffle=False, 
                                                     batch_size=1,
                                                     class_mode=None) # no targets get yielded (only input images are yielded).
            logger.info("image generator finish.")
    
            logger.info("begin to predict gen data")
            test = model.predict_generator(test_generator, steps = test_generator.samples, verbose = 1)
            logger.info(test.shape)
            with h5py.File("../data/model/gap_{}_{}.h5".format(MODEL.func_name,version),'w') as h:
                h.create_dataset("gen", data=test)


    def process(self):

        for m in self.use_model:
            logger.info("Begin to process {} model".format(m))

            if m == "ResNet50":
                self.genration(ResNet50,(224,224))

            elif m == "Xception":
                self.genration(Xception,(299,299), xception_input)

            elif m == "InceptionV3":
                self.genration(InceptionV3,(299,299), inception_input)

            elif m == "VGG16":
                self.genration(VGG16,(224,224))

            elif m == "VGG19":
                self.genration(VGG19,(224,224))
            else:
                logger.error("{} model not found!".format(m))



if __name__ == "__main__":
    
    ##############################
    # CNF GET
    ##############################

    CNF = yaml.load(open('../conf/setting.yaml'))
    
    version = CNF['version']

    prefix = CNF['prefix']

    use_model = CNF['use_model']
    gen_layer = CNF['gen_layer']

    genPath = CNF['gen_path']
    train_path = prefix + CNF['train_path']
    test_path = prefix + CNF['test_path']
    log_path = prefix + CNF['log_path']
    if not os.path.isdir(train_path): os.mkdir(train_path)
    if not os.path.isdir(test_path): os.mkdir(test_path)
    if not os.path.isdir(log_path): os.mkdir(log_path)

    logger =  myLog(log_path+"/log_{}".format(version))  
    logger.info(CNF)

    gener = feature_generation(
        trainPath = train_path,
        testPath = test_path,
        version = version,
        genPath = genPath,
        gen_layer = gen_layer,
        use_model = use_model
        )
    gener.process()

#write_gap(Xception, (299, 299), xception_input)

#write_gap(InceptionV3, (299, 299), inception_input)

#write_gap(VGG16, (224, 224))

#write_gap(VGG19, (224, 224))

#Your statements here

stop = timeit.default_timer()

print stop - start 
