#encoding=utf-8

import argparse

parser = argparse.ArgumentParser(description="get index of pics for train and validation & resize pics and save")
parser.add_argument('-rs','--resize',acrion='store',type=int,nargs='+',
        default=[227,227],help='resize height and width')
parser.add_argument('-ip','--origin_pic_path',action='store',type=int,
        default='../../data/pic_car_backup/images',help='resize height and width')
parser.add_argument('-op','--output_pic_path',action='store',type=int,
        default='../../data/pic_car_backup/images',help='resize height and width')
parser.add_argument('-op','--output_pic_path',action='store',type=int,
        default='../../data/pic_car_backup/images',help='resize height and width')


args = parser.parse_args()
print("\r="*50)
print("[INFO] args:\r")
print(args)
print("\r="*50)


