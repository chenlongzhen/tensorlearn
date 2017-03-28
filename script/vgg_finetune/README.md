# vgg
vggnet version1

# 代码
环境：python3 keras2(tensorflow backend)
## 结构
name | usage
---|---
data_utils | 参数文件下载等，vgg16调用
vgg16.py | vgg核心代码
vgg16_fintune.py|vgg finetune
vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5 | vgg 默认参数，需要放在 ~/.keras/models下
cnn_dog_cat_demo.py | keras cnn基础例子

## 调用
```
python vgg16_fintune.py
```

## dogVScat 例子
1. 图片文件结构如下
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
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
```

2. vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5 | vgg 默认参数，需要放在 ~/.keras/models下
3. python vgg16_fintune.py
4. 参数设置python vgg16_fintune.py -h 


# vgg 结构
![image](http://note.youdao.com/yws/public/resource/696b244f911b9707b9a3e70050d34cad/xmlnote/31E24F1F40B04F109DD9F5535C25432E/6369)

# Alex的结构
![image](http://note.youdao.com/yws/public/resource/696b244f911b9707b9a3e70050d34cad/xmlnote/C437476CB1094793A80C71DDE8D20FCA/6372)

# 经过每一层图片的变化

![image](http://note.youdao.com/yws/public/resource/696b244f911b9707b9a3e70050d34cad/xmlnote/74A2C350DC9843B5ACCD563F1DE6AE4D/6374)

# reference
[1]. https://jacobgil.github.io/deeplearning/filter-visualizations
[2]. https://nbviewer.jupyter.org/gist/embanner/6149bba89c174af3bfd69537b72bca74
[3]. https://icmlviz.github.io/assets/papers/4.pdf
