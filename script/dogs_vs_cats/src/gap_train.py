
# coding: utf-8

# In[1]:

import h5py
import sys,os

import logging
import numpy as np
from keras.models import save_model
from sklearn.utils import shuffle
import yaml
def myLog(logPath):
    '''
    logging
    :param logPath:  where to save log
    :return: logging handle
    '''

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


#################################
# cofig
#################################
CNF = yaml.load(open('../conf/setting.yaml'))

version = CNF['version']

test_path = CNF['test_path']

use_model = CNF['use_model']
gen_layer = CNF['gen_layer']

log_path = CNF['log_path']

np.random.seed(int(CNF['seed']))

logger = myLog(log_path+"/log_{}".format(version))
logger.info(CNF)

submissionPath =  "../data/output/submission.csv"
END_MODEL =  "../data/endModel/endModel_{}.h5".format(version)

X_train = []
X_test = []


for m in use_model:
    filename =  "../data/model/gap_{}_{}.h5".format(m,version)
    logger.info("[INFO] begin to read {}".format(os.path.basename(filename)))
    with h5py.File(filename, 'r') as h:
        X_train.append(np.array(h['train']))
        X_test.append(np.array(h['test']))
        y_train = np.array(h['label'])

X_train = np.concatenate(X_train, axis=1)
X_test = np.concatenate(X_test, axis=1)

logger.info("train shape\r")
logger.info(X_train.shape)
logger.info("test shape\r")
logger.info(X_test.shape)

logger.info("[INFO] shuffle")
X_train, y_train = shuffle(X_train, y_train)



from keras.models import *
from keras.layers import *

logger.info("[INFO] train")
X_train, y_train = shuffle(X_train, y_train)
input_tensor = Input(X_train.shape[1:])
x = input_tensor
x = Dropout(0.5)(x)
x = Dense(1, activation='sigmoid')(x)
model = Model(input_tensor, x)
model.summary()

model.compile(optimizer='adadelta',
              loss='binary_crossentropy',
              metrics=['accuracy'])



## In[3]:
#
#from IPython.display import SVG
#from keras.utils.visualize_util import model_to_dot, plot
#
#SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))

model.fit(X_train, y_train, batch_size=128, nb_epoch=8, validation_split=0.2)


logger.info("[INFO] save model")
save_model(model,END_MODEL)


# In[6]:

# 预测 
print("[INFO] predict")
y_pred = model.predict(X_test, verbose=1)
y_pred = y_pred.clip(min=0.005, max=0.995)


# In[7]:
# 找回原来的文件名并和预测值保存    
print("[INFO] save predict")
import pandas as pd
from keras.preprocessing.image import *

df = pd.read_csv("../data/output/sample_submission.csv")


image_size = (224, 224)
gen = ImageDataGenerator()
test_generator = gen.flow_from_directory(test_path, image_size, shuffle=False,
                                         batch_size=1, class_mode=None)


for i, fname in enumerate(test_generator.filenames):
    index = int(fname[fname.rfind('/')+1:fname.rfind('.')])
    df.set_value(index-1, 'label', y_pred[i])

df.to_csv(submissionPath, index=None)
df.head(10)
