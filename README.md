# mxnet-style-transfer
基于mxnet 的图片风格迁移


### 1. 参考博客
- [风格迁移研究概述(机器之心)](https://www.jiqizhixin.com/articles/2017-05-15-3)

- [谈谈图像的Style Transfer（一）](http://blog.csdn.net/hungryof/article/details/53981959)

- [谈谈图像的Style Transfer（二）](http://blog.csdn.net/hungryof/article/details/71512406)
- [利用卷积神经网络实现图像风格迁移 (一)](http://blog.csdn.net/matrix_space/article/details/54286086)
- [利用卷积神经网络实现图像风格迁移 (二)](http://blog.csdn.net/matrix_space/article/details/54290460)
- [图像风格转换(Image style transfer)](http://dataunion.org/26797.html)




### 2. 相关论文
- [Image Style Transfer Using Convolutional Neural Networks （中文版）](http://blog.csdn.net/cicibabe/article/details/70885715)

- [Deep Photo Style Transfer (中文版)](http://blog.csdn.net/cicibabe/article/details/70868746)




### 3. Image Style Transfer 实现

#### 3.1 VGG-19
> 论文使用VGG-19网络模型提取图片的style及content。
>
> 因此，首先需要构建一个**已训练**好的VGG-19 网络模型

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

  ​

- VGG(
    (features): HybridSequential(

      (0): Conv2D(3 -> 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): Activation(relu)
      (2): Conv2D(64 -> 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (3): Activation(relu)
      (4): MaxPool2D(size=(2, 2), stride=(2, 2), padding=(0, 0), ceil_mode=False)
      (5): Conv2D(64 -> 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (6): Activation(relu)
      (7): Conv2D(128 -> 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (8): Activation(relu)
      (9): MaxPool2D(size=(2, 2), stride=(2, 2), padding=(0, 0), ceil_mode=False)
      (10): Conv2D(128 -> 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (11): Activation(relu)
      (12): Conv2D(256 -> 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (13): Activation(relu)
      (14): Conv2D(256 -> 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (15): Activation(relu)
      (16): Conv2D(256 -> 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (17): Activation(relu)
      (18): MaxPool2D(size=(2, 2), stride=(2, 2), padding=(0, 0), ceil_mode=False)
      (19): Conv2D(256 -> 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (20): Activation(relu)
      (21): Conv2D(512 -> 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (22): Activation(relu)
      (23): Conv2D(512 -> 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (24): Activation(relu)
      (25): Conv2D(512 -> 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (26): Activation(relu)
      (27): MaxPool2D(size=(2, 2), stride=(2, 2), padding=(0, 0), ceil_mode=False)
      (28): Conv2D(512 -> 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (29): Activation(relu)
      (30): Conv2D(512 -> 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (31): Activation(relu)
      (32): Conv2D(512 -> 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (33): Activation(relu)
      (34): Conv2D(512 -> 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (35): Activation(relu)
      (36): MaxPool2D(size=(2, 2), stride=(2, 2), padding=(0, 0), ceil_mode=False)
      (37): Dense(25088 -> 4096, Activation(relu))
      (38): Dropout(p = 0.5)
      (39): Dense(4096 -> 4096, Activation(relu))
      (40): Dropout(p = 0.5)
    )
    (output): Dense(4096 -> 1000, linear)
  )


  ​