#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/25 17:04
# @Author  : Leishichi
# @File    : image_style_transfer_model.py
# @Software: PyCharm
# @Tag:

import mxnet as mx
from mxnet import gluon
from mxnet.gluon.model_zoo import vision
from src.tool import net_tool as tool


def style_net(vgg):
    """
    定义用于提取风格特征的网络
    提取VGG-19 中下列卷积层的输出作为 风格特征值
    conv1_1 , conv2_1 , conv3_1 , conv4_1 , conv5_1
    :param vgg:  mxnet.gluon.model_zoo.vision.vgg19_bn
    :return:
    """
    # 获取要输出的层对应symbol列表
    data = mx.sym.var('data')
    internals = vgg(data).get_internals()
    list = [internals['vgg0_conv0_fwd_output'],
            internals['vgg0_conv2_fwd_output'],
            internals['vgg0_conv4_fwd_output'],
            internals['vgg0_conv8_fwd_output'],
            internals['vgg0_conv12_fwd_output']]

    # 根据原有网络构建新的网络
    net = gluon.SymbolBlock(list, data, params=vgg.collect_params())
    return net


def content_net(vgg):
    """
    定义用于提取内容特征的网络
    提取VGG-19 中下列卷积层的输出作为 风格特征值
    conv4_2
    :param vgg:  mxnet.gluon.model_zoo.vision.vgg19_bn
    :return:
    """
    # 获取要输出的层对应symbol列表
    data = mx.sym.var('data')
    internals = vgg(data).get_internals()
    list = [internals['vgg0_conv9_fwd_output']]

    # 根据原有网络构建新的网络
    net = gluon.SymbolBlock(list, data, params=vgg.collect_params())
    return net


def net():
    pass
