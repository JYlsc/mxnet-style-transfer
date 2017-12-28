#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/25 17:04
# @Author  : Leishichi
# @File    : model.py
# @Software: PyCharm
# @Tag:

import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import ndarray as nd


class FeatureNet():
    def __init__(self, vgg):
        """
        初始化特征网络
        :param vgg: 使用VGG_19构建特征网络
        """
        self.net = nn.Sequential()
        # 需要更改的pool层index
        pooling = {4, 9, 18}
        for i in range(29):
            if (i in pooling):
                # 将MaxPool改为AvgPool
                self.net.add(nn.AvgPool2D(strides=(2, 2)))
            else:
                self.net.add(vgg.features[i])

    def get_features(self, x):
        style_layers = [0, 5, 10, 19, 28]
        content_layers = [21]
        style = []
        content = _
        for i in range(len(self.net)):
            x = self.net[i](x)
            if i in style_layers:
                style.append(x)
            if i in content_layers:
                content = x
        return content, style

    def get_loss(self, x, y):
        """
        计算loss
        :param x:
        :param y:
        :return:
        """
        return (x - y).square().mean()

    def get_grams(self, features):
        list = []
        for feature in features:
            list.append(self.get_gram(feature))
        return list

    def get_gram(self, features):
        """
        计算features 的 gram矩阵
        :param features:
        :return:
        """
        c = features.shape[1]
        n = features.size / features.shape[1]
        y = features.reshape((c, int(n)))
        return nd.dot(y, y.T) / n

    def get_style_loss(self, grams, _grams):
        """
        获取风格loss
        :param grams:
        :param _grams:
        :return:
        """
        loss = 0.
        for i in range(len(grams)):
            # 计算风格loss，越靠近输入层的gram权重越高
            loss = loss + self.get_loss(grams[i], _grams[i]) / i
        return loss


    def get_tv_loss(self,x):
        """
        总变差降噪（Total Variation Denoising)
        去除噪音
        :return:
        """
        return 0.5 * ((x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean() +
                      (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean())

#
#
# # 另外一种获取中间层输出的方法
# def features_net(vgg):
#     """
#     使用 SymbolBlock 获取指定中间层的输出
#     :param vgg:
#     :return:
#     """
#     # 获取要输出的层对应symbol列表
#     data = mx.sym.var('data')
#     internals = vgg(data).get_internals()
#     # print(internals.list_outputs())
#     list = [internals['vgg0_conv9_fwd_output'],
#             internals['vgg0_conv0_fwd_output'],
#             internals['vgg0_conv2_fwd_output'],
#             internals['vgg0_conv4_fwd_output'],
#             internals['vgg0_conv8_fwd_output'],
#             internals['vgg0_conv12_fwd_output']]
#
#     # 根据原有网络构建新的网络
#     net = gluon.SymbolBlock(list, data, params=vgg.collect_params())
#     return net
