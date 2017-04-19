# coding: utf-8
# 用于生成制定格式的输入样本,  软连接形式
# author:chenlongzhen

import os
#import shutil
import sys

# In[2]:
origin_path = os.path.abspath(sys.argv[1])
train_file_list = os.path.abspath(sys.argv[2])
out_path = os.path.abspath(sys.argv[3])


#train_filenames = os.listdir('train')

def mkdir(dir):
    if not os.path.isdir(dir):
        print("[INFO] mkdir {}".format(dir))
        os.mkdir(dir)


mkdir(out_path)
with open(train_file_list,'r') as rfile:
    for num,line in enumerate(rfile):
        if num % 10000 == 1:
            print("[INFO] {} lines processed".format(num-1))
        segs = line.strip().split(',')
        id = segs[0]
        label = segs[1]
        mkdir(out_path+"/"+label)
        os.symlink(origin_path+'/'+  id + ".png",out_path+"/"+label+"/"+id + ".png")



#train_cat = filter(lambda x:x[:3] == 'cat', train_filenames)
#train_dog = filter(lambda x:x[:3] == 'dog', train_filenames)


# In[3]:

#def rmrf_mkdir(dirname):
#    if os.path.exists(dirname):
#        shutil.rmtree(dirname)
#    os.mkdir(dirname)

#os.symlink('../test/', 'test2/test')
#
#for filename in train_cat:
#    os.symlink('../../train/'+filename, 'train2/cat/'+filename)
#
#for filename in train_dog:
#    os.symlink('../../train/'+filename, 'train2/dog/'+filename)





