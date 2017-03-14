#encoding=utf-8

import urllib.request,sys,os,random
S_PATH = sys.path[0]
DATA_PATH=S_PATH+"/../../data"

input_name=sys.argv[1]
pic_path = sys.argv[2]
ratio = float(sys.argv[3])

base_dir = DATA_PATH+"/base_data"
pic_path_abs = DATA_PATH+"/"+pic_path 

input_file = DATA_PATH+"/base_data/"+input_name
output_file_train = DATA_PATH+"/base_data/car_train.txt"
output_file_test = DATA_PATH+"/base_data/car_test.txt"

if not os.path.isdir(base_dir):os.mkdir(base_dir)
if not os.path.isdir(pic_path_abs):os.mkdir(pic_path_abs)

series_col=1
pic_col=3

with open(input_file,'r') as rfile,open(output_file_train,'w') as trainfile,open(output_file_test,'w') as testfile:
    all_num = 0
    count=0
    for line in rfile:
        all_num += 1
        segs = line.split('\t')
        serie = segs[series_col]
        pic_url = segs[pic_col]
        print(pic_col)
        if pic_url.strip().split(".")[-1] == 'jpg':
            tail = ".jpg"
        elif pic_url.strip().split(".")[-1] == 'png':
            tail = ".png"
        else:
            print( pic_url.split(".")[-1] )
            continue 
        try:
            print("{}\tdownloading...".format(pic_url))
            save_path= DATA_PATH + "/" +pic_path + "/"+str(count)+tail
            print("save:" +save_path)
            urllib.request.urlretrieve(pic_url,save_path)
            if random.random() <= ratio:
                trainfile.write("{} {}\n".format(save_path,serie)) 
            else:
                testfile.write("{} {}\n".format(save_path,serie)) 
            count+=1
        except Exception as e:
            print('【错误】当前图片无法下载\n{}'.format(e))
            continue
    print("all pic {}. {} pic successfullu download.".format(all_num,count))

#python get_pic.py base_pic_data images 0.8
