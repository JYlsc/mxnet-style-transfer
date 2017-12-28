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

from src.main.image_style_transfer.model import FeatureNet
from src.tool import net_tool as tool
from src.main.image_style_transfer import loss as loss_function

# 判读是否使用gpu
ctx = tool.get_ctx()

# 内容风格占比,当觉得风格不明显时可以放大该值
alpha = 5000

# 噪音消除的权重
beta = 10

# 学习速率,需按实际情况进行调整
# 每张图片可能都不一样
learning_rate = 1

# 迭代次数
iter = 10000

# 生成图片大小
# GTX1060 3G,尼玛只能生成400*400的图！！！
img_size = 400

data_path = "../../../data/"

# 设置输入输出文件路径
input_path = data_path + "img/content/input.jpg"
style_path = data_path + "img/style/style7.jpg"
output_path = data_path + "img/output/"

# vgg 参数路径
param = data_path + "param/"


def style_transfer(net, content_img, style_img):
    # 获取style及content
    content, _ = net.get_features(content_img)
    _, style = net.get_features(style_img)

    # 提前计算风格图的gram已提高计算速度
    grams = net.get_grams(style)

    # 构建储存输出图片的Parameter
    result = gluon.Parameter('temp', shape=content_img.shape)
    result.initialize(ctx=ctx)
    result.set_data(content_img)

    # 构建训练器
    trainer = gluon.Trainer([result], 'adam', {'learning_rate': learning_rate})

    # 迭代生成新图片
    for e in range(iter):
        with autograd.record():
            # 获取目标图片当前的风格及内容特征
            _content, _style = net.get_features(result.data())
            _grams = net.get_grams(_style)

            # 计算总loss
            loss = net.get_loss(content, _content) \
                   + alpha * net.get_style_loss(grams, _grams) \
                   + beta * net.get_tv_loss(result.data())

        loss.backward()
        trainer.step(1)

        print("次数:", e, "  loss:", loss)

        if e % 100 == 0:
            tool.save_img(result.data(), output_path + str(e) + ".png")

    return result.data()


if __name__ == "__main__":
    # 加载vgg模型
    vgg = vision.vgg19_bn(ctx=ctx, pretrained=True, root=param)

    # 构建风格网络及内容网络
    net = FeatureNet(vgg)

    # 读取图片
    content_img = tool.read_img(input_path, ctx)
    style_img = tool.read_img(input_path, ctx)

    result_img = style_transfer(net, content_img, style_img)

    # 储存结果
    tool.save_img(result_img, output_path + "result.png")
