#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/22 16:35
# @Author  : Leishichi
# @File    : main.py
# @Software: PyCharm
# @Tag: 《 Image style transfer using convolutional neural networks 》 基于mxnet的实现
from mxnet import gluon, autograd
import mxnet as mx
from mxnet.gluon.model_zoo import vision
from src.main.image_style_transfer import model as model
from src.main.image_style_transfer.model import FeatureNet
from src.tool import net_tool as tool

# 判读是否使用gpu
ctx = tool.get_ctx()

# 内容风格占比，值越大，风格越突出
alpha = 5000

# 学习速率，需根据实际情况进行调节
# 同时学习速率也可以视为画笔大小，速率越大图片风格越粗旷
learning_rate = 10

# tv_loss 权重，值越大图片越干净
beta = 10

# 迭代次数
iter = 1000

# 生成的图片大小
size = 400

data_path = "../../../data/"

# 设置输入输出文件路径
input_path = data_path + "img/content/dog.jpg"
style_path = data_path + "img/style/head.jpg"
output_path = data_path + "img/output/"

# vgg 参数路径
param = data_path + "param/"


def style_transfer(net, content_img, style_img):
    # 获取style及content
    content = net.get_feature(content_img)[0]
    style = net.get_feature(style_img)[1:]

    output = gluon.Parameter('_img', shape=content_img.shape)
    output.initialize(ctx=ctx)
    output.set_data(content_img)

    trainer = gluon.Trainer([output], 'adam', {'learning_rate': learning_rate})

    # 迭代获取新图片
    for e in range(iter):
        with autograd.record():
            _img = output.data()
            # 获取content 及style
            _features = net.get_feature(_img)
            _content = _features[0]
            _style = _features[1:]
            # 计算风格loss
            style_loss = alpha * net.get_style_loss(_style, style)
            # 计算内容loss
            content_loss = net.get_loss(content, _content)
            # 计算总变差降噪loss
            tv_loss = beta * net.get_tv_loss(_img)
            loss = style_loss + content_loss + tv_loss

        loss.backward()
        trainer.step(1)
        print("-" * 30)
        print("tv_loss:", tv_loss,
              "\ncontent_loss:", content_loss,
              "\nstyle_loss:", style_loss,
              "\n迭代次数:", e,
              "\nloss:",loss)

        if e % 100 == 0:
            tool.save_img(output.data(), output_path + str(e) + ".png")
    tool.save_img(output.data(), output_path + "result.png")


if __name__ == "__main__":
    # 加载vgg模型
    vgg = vision.vgg19(ctx=ctx, pretrained=True, root=param)

    # 构建风格网络及内容网络
    net = FeatureNet(vgg)

    # 读取图片
    content_img = tool.read_img(input_path, ctx=ctx, size=size)
    style_img = tool.read_img(style_path, ctx=ctx, size=size)

    # 生成图片
    style_transfer(net, content_img, style_img)
