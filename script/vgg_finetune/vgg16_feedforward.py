'''
@clz
model feedforward prediction
'''

from PIL import Image
from keras.models import save_model
from keras.models import load_model
import numpy as np
import sys
import glob
import getImage4Predict

model_path = sys.argv[1]
pic_path = sys.argv[2]
# load model
model = load_model(model_path)

#
#def get_pic(path):
#    pathList = glob.glob(path+'/*')
#
#
#im = Image.open(pic_path).resize((224, 224), Image.ANTIALIAS)
#im = np.array(im).astype(np.float32) # 2array
#
## scale the image, according to the format used in training
#im[:,:,0] -= 103.939
#im[:,:,1] -= 116.779
#im[:,:,2] -= 123.68
#im = im.transpose((1,0,2))
#im = np.expand_dims(im, axis=0)
#print(im.shape)


def save(files,out,path='output/out.csv'):
    with open(path,'w') as inf:
        for filename,out in zip(files,out):
            print("{},{}".format(filename,out[0]))
            inf.write("{},{}\n".format(filename,out[0]))

files,ims = getImage4Predict.getPics(pic_path)
print("datashape:{}".format(ims.shape))

# out
out = model.predict(ims)
print("files:{}".format(files))
print("inference:{}".format(out))

save(files,out)
