
# coding: utf-8

# In[1]:

import h5py
import sys,os
import numpy as np
from keras.models import save_model
from sklearn.utils import shuffle
np.random.seed(2017)

S_PATH = sys.path[0]
DATA_PATH = S_PATH + "/../data/"

gap_ResNet50 = DATA_PATH + "gap_ResNet50.h5"
gap_Xception = DATA_PATH + "gap_Xception.h5"
gap_InceptionV3 = DATA_PATH + "gap_InceptionV3.h5"
END_MODEL = DATA_PATH + "/endModel.h5"

test_data = DATA_PATH + "../../vgg_finetune/data/testPic"

submissionPath = DATA_PATH + "/submission.csv"

X_train = []
X_test = []


for filename in [gap_ResNet50, gap_Xception, gap_InceptionV3]:
    print("[INFO] begin to read {}".format(os.path.basename(filename)))
    with h5py.File(filename, 'r') as h:
        X_train.append(np.array(h['train']))
        X_test.append(np.array(h['test']))
        y_train = np.array(h['label'])

X_train = np.concatenate(X_train, axis=1)
X_test = np.concatenate(X_test, axis=1)
print(X_train.shape)
print(X_test.shape)

print("[INFO] shuffle")
X_train, y_train = shuffle(X_train, y_train)


# In[2]:

from keras.models import *
from keras.layers import *

print("[INFO] train")
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


# In[4]:

model.fit(X_train, y_train, batch_size=128, nb_epoch=8, validation_split=0.2)


# In[5]:

print("[INFO] save model")
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

df = pd.read_csv(DATA_PATH + "/sample_submission.csv")


image_size = (224, 224)
gen = ImageDataGenerator()
test_generator = gen.flow_from_directory(test_data, image_size, shuffle=False, 
                                         batch_size=1, class_mode=None)


for i, fname in enumerate(test_generator.filenames):
    index = int(fname[fname.rfind('/')+1:fname.rfind('.')])
    df.set_value(index-1, 'label', y_pred[i])

df.to_csv(submissionPath, index=None)
df.head(10)
