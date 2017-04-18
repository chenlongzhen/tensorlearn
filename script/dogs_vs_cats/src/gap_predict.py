
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

pre_fix = CNF['prefix']

test_path = CNF['test_path']

use_model = CNF['use_model']
gen_layer = CNF['gen_layer']

log_path = pre_fix + '/' +CNF['log_path']

dog_cat = CNF['dog_cat']

np.random.seed(int(CNF['seed']))

logger = myLog(log_path+"/log_{}".format(version))
logger.info(CNF)

submissionPath =  "../data/output/submission.csv"
END_MODEL =  "../data/endModel/endModel_{}.h5".format(version)

save_path = pre_fix + '/' + CNF['save_path']

#################################
# predict
#################################
X_train = []
X_test = []

from keras.models import *

print("[INFO] load model")
model = load_model(END_MODEL)

print("[INFO] predict")
y_pred = model.predict(X_test, verbose=1)



#################################
# 找回原来的文件名并和预测值保存
#################################
print("[INFO] save predict")
from keras.preprocessing.image import *

gen = ImageDataGenerator()
test_generator = gen.flow_from_directory(test_path, (224,224), shuffle=False,
                                         batch_size=1, class_mode=None)

if dog_cat == 1:
    import pandas as pd
    y_pred = y_pred.clip(min=0.005, max=0.995)
    df = pd.read_csv("../data/output/sample_submission.csv")
    for i, fname in enumerate(test_generator.filenames):
        index = int(fname[fname.rfind('/')+1:fname.rfind('.')])
        df.set_value(index-1, 'label', y_pred[i])

    df.to_csv(submissionPath, index=None)
    df.head(10)

else:
    with open(save_path,'w') as ifile:
        for i, fname in enumerate(test_generator.filenames):
            ifile.write("{},{}\n".format(fname,pre_fix[i]))










