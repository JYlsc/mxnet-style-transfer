#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/22 16:35
# @Author  : Leishichi
# @File    : image_style_transfer.py
# @Software: PyCharm
# @Tag: 《 Image style transfer using convolutional neural networks 》 基于mxnet的实现
from mxnet import gluon, autograd
import mxnet as mx
from mxnet.gluon import Parameter
from mxnet.gluon.model_zoo import vision
from src.main import image_style_transfer_model as model
from src.tool import net_tool as tool

# 判读是否使用gpu
ctx = tool.get_ctx()

content_weight = 1
style_weight = 100
learning_rate = 0.01


def content_loss(x, y):
    """
    内容loss function
    :param x:
    :param y:
    :return:
    """
    return gluon.loss.L2Loss()(x, y)


def gram(features):
    """
    计算features 的 gram矩阵
    :param features:
    :return:
    """
    (b, ch, h, w) = features.shape
    features = features.reshape((b, ch, w * h))
    gram = mx.ndarray.batch_dot(features, features, transpose_b=True)
    return gram


def style_loss(x_list, y_list):
    loss = 0.
    for i in range(len(x_list)):
        x = x_list[i]
        y = y_list[i]
        shape = x.shape
        loss_function = gluon.loss.L2Loss()
        M = shape[2] * shape[3]
        N = shape[1]

        A = gram(y)
        G = gram(x)
        loss = loss + loss_function(A, G) * (1. / (2 * (N ** 2) * (M ** 2)))
    return loss


def train():
    # 加载vgg模型
    param = "../../data/param/"
    vgg = vision.vgg19(ctx=ctx, pretrained=True, root=param)

    # 构建风格网络及内容网络
    features_net = model.features_net(vgg)

    # 读取图片
    input_img, style_img = read_input()

    # 获取style及content
    features = features_net(input_img)



    output = gluon.Parameter('_img', shape=input_img.shape)
    output.initialize(ctx=ctx)
    output.set_data(tool.new_img(input_img.shape))

    trainer = gluon.Trainer([output], 'adam', {'learning_rate': learning_rate})

    # 迭代获取新图片
    for e in range(10000):
        with autograd.record():
            _img = output.data()
            _features = features_net(_img)
            loss = content_weight * content_loss(features[0], _features[0])
        print("次数:", e, "  loss:", loss)
        loss.backward()
        trainer.step(1)
        if e % 100 == 0:
            tool.save_img(output.data(), "../../data/img/output" + str(e) + ".png")
    tool.save_img(output.data(), "../../data/img/output.png")


def read_input():
    # 设置输入输出文件路径
    input_path = "../../data/img/input.jpg"
    style_path = "../../data/img/style.jpg"

    # 读取图片
    input_img = tool.read_img(input_path).as_in_context(ctx)
    style_img = tool.read_img(style_path).as_in_context(ctx)
    return input_img, style_img


train()
