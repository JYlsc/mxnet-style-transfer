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

content_weight = 0.8
style_weight = 0.8
learning_rate = 0.01


def content_loss(x, y):
    loss_function = gluon.loss.L2Loss()
    loss = loss_function(x, y)
    return loss


def gram(x, m, n):
    y = x.reshape((m,n))
    gram = mx.ndarray.dot(y.T, y)
    return gram

def gram_matrix(y):
    (b, ch, h, w) = y.shape
    features = y.reshape((b, ch, w * h))
    #features_t = F.SwapAxis(features,1, 2)
    gram = mx.ndarray.batch_dot(features, features, transpose_b=True) / (ch * h * w)
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

        A = gram_matrix(y)
        G = gram_matrix(x)
        loss = (1. / (4 * N ** 2 * M ** 2)) * loss_function(A, G)+loss
    return loss


def train():
    # 加载vgg模型
    param = "../../data/param/"
    vgg = vision.vgg19_bn(ctx=ctx, pretrained=True, root=param)

    # 构建风格网络及内容网络
    style_net = model.style_net(vgg)
    content_net = model.content_net(vgg)

    # 读取图片
    input_img, style_img = read_input()

    # 获取style及content
    style = style_net(style_img)
    content = content_net(input_img)

    # output
    output = Parameter('output', shape=input_img.shape)
    output.initialize(ctx=ctx)
    output.set_data(input_img)

    trainer = gluon.Trainer([output], 'adam', {'learning_rate': learning_rate})

    # 迭代获取新图片
    for e in range(200):
        with autograd.record():
            y_style = style_net(output.data())
            y_content = content_net(output.data())
            loss = content_weight * content_loss(y_content, content) + style_weight * style_loss(y_style, style)
        print("次数:", e, "  loss:", loss)
        loss.backward()
        trainer.step(1)
    tool.save_img(output.data(), "../../data/img/output.png")


def read_input():
    # 设置输入输出文件路径
    input_path = "../../data/img/input.jpg"
    style_path = "../../data/img/style.jpg"

    # 读取图片
    input_img = tool.read_img(input_path)
    style_img = tool.read_img(style_path)
    return input_img, style_img


train()
