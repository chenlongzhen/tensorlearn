# CNN 
支持
    - ResNet50
    - Xception
    - InceptionV3
    - VGG16 
特征生成，和融合训练预测

# 代码
环境：python2 keras2(tensorflow backend)
## 结构
name | usage
---|---
gap.py | 生成特征并保存
gap_train.py | 读取生成的特征训练,保存模型
gap_predict.py| 读取模型和gap.py生成的特征预测 
Preprocessing train dataset gap.py | 构造数据集工具,需要根据不同情况修改


### 图片文件结构如下
```
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    test/
        test/
            001.jpg
            002.jpg
            004.jpg
            005.jpg
```

# conf 参数
见conf文件

# 注意
需要预先准备好权重文件,否则代码自己下载很慢：
链接: https://pan.baidu.com/s/1dFchLOt 密码: dxxu
放在 ~/.keras/models/ 
