# mxnet-style-transfer
基于mxnet 的图片风格迁移


### 1. 参考博客
- [风格迁移研究概述(机器之心)](https://www.jiqizhixin.com/articles/2017-05-15-3)

- [谈谈图像的Style Transfer（一）](http://blog.csdn.net/hungryof/article/details/53981959)

- [谈谈图像的Style Transfer（二）](http://blog.csdn.net/hungryof/article/details/71512406)
- [利用卷积神经网络实现图像风格迁移 (一)](http://blog.csdn.net/matrix_space/article/details/54286086)
- [利用卷积神经网络实现图像风格迁移 (二)](http://blog.csdn.net/matrix_space/article/details/54290460)
- [图像风格转换(Image style transfer)](http://dataunion.org/26797.html)
- [李沐：样式迁移](http://zh.gluon.ai/chapter_computer-vision/neural-style.html)




### 2. 相关论文

- [Image Style Transfer Using Convolutional Neural Networks （中文版）](http://blog.csdn.net/cicibabe/article/details/70885715)

- [Deep Photo Style Transfer (中文版)](http://blog.csdn.net/cicibabe/article/details/70868746)




### 3. Image Style Transfer 实现

#### 3.1 VGG-19
> 论文使用VGG-19网络模型提取图片的style及content。
>
> 因此，首先需要构建一个**已训练**好的VGG-19 网络模型
>
> VGG 参数可手动下载放入  /data/param/   目录下

| VGG-19 网络结构                             | 层数     |
| :-------------------------------------- | ------ |
| input (224 × 224 RGB image)             |        |
| conv3-64，conv3-64                       | conv 1 |
| avgpool                                 |        |
| conv3-128 ，conv3-128                    | conv 2 |
| avgpool                                 |        |
| conv3-256，conv3-256，conv3-256，conv3-256 | conv 3 |
| avgpool                                 |        |
| conv3-512，conv3-512，conv3-512，conv3-512 | conv 4 |
| avgpool                                 |        |
| conv3-512，conv3-512，conv3-512，conv3-512 | conv 5 |
| avgpool                                 |        |
| FC-4096                                 |        |
| FC-4096                                 |        |
| FC-1000                                 |        |
| soft-max                                |        |


- Content : conv 4-2


- Style  :  conv1-1 , conv2-1 , conv3-1 , conv4-1 , conv5-1




#### 3.2 Content loss

 计算内容的loss函数直接使用L2Loss 即可



#### 3.3 Style Loss

- 首先计算每个feature 的Gram矩阵
- 对Gram矩阵计算L2Loss 



#### 3.4 Loss

对content_loss 和 style_loss 进行加权求和



