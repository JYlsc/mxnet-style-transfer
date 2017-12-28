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
from src.tool import net_tool as tool
from src.main.image_style_transfer import loss as loss_function

# 判读是否使用gpu
ctx = tool.get_ctx()

# 内容风格占比
alpha = 0.05
# 学习速率
learning_rate = 0.01

# 设置输入输出文件路径
input_path = "../../../data/img/content/input.jpg"
style_path = "../../../data/img/style/style6.jpg"


def train():
    # 加载vgg模型
    param = "../../data/param/"
    vgg = vision.vgg19(ctx=ctx, pretrained=True, root=param)

    # 构建风格网络及内容网络
    features_net = model.features_net(vgg)

    # 读取图片
    img, style_img = read_input()

    # 获取style及content
    content = features_net(img)[0]
    style = features_net(style_img)[1:]

    output = gluon.Parameter('_img', shape=img.shape)
    output.initialize(ctx=ctx)
    output.set_data(img)

    tool.save_img(output.data(), "../../data/img/src.jpg")
    trainer = gluon.Trainer([output], 'adam', {'learning_rate': learning_rate})

    # 迭代获取新图片
    for e in range(1000):
        with autograd.record():
            _img = output.data()
            _features = features_net(_img)
            _content = _features[0]
            _style = _features[1:]
            loss = loss_function.loss(content,_content,style,_style,alpha)

        loss.backward()
        trainer.step(1)

        print("次数:", e, "  loss:", loss)

        if e % 100 == 0:
            tool.save_img(output.data(), "../../data/img/output" + str(e) + ".png")
    tool.save_img(output.data(), "../../data/img/output.png")


def read_input():
    # 读取图片
    input_img = tool.read_img(input_path).as_in_context(ctx)
    style_img = tool.read_img(style_path).as_in_context(ctx)
    return input_img, style_img


train()
