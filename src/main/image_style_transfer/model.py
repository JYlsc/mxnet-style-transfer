#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/25 17:04
# @Author  : Leishichi
# @File    : model.py
# @Software: PyCharm
# @Tag:

import mxnet as mx
from mxnet import gluon
from mxnet import ndarray as nd


class FeatureNet():

    def __init__(self, vgg):
        self.net = self.init_net(vgg)
        pass

    def init_net(self, vgg):
        # 获取要输出的层对应symbol列表
        data = mx.sym.var('data')
        internals = vgg(data).get_internals()
        print(internals.list_outputs())
        list = [internals['vgg0_conv9_fwd_output'],
                internals['vgg0_conv0_fwd_output'],
                internals['vgg0_conv2_fwd_output'],
                internals['vgg0_conv4_fwd_output'],
                internals['vgg0_conv8_fwd_output'],
                internals['vgg0_conv12_fwd_output']]

        # 根据原有网络构建新的网络
        net = gluon.SymbolBlock(list, data, params=vgg.collect_params())
        return net

    def get_gram(self, feature):
        """
        计算features 的 gram矩阵
        :param features:
        :return:
        """
        (b, ch, h, w) = feature.shape
        features = feature.reshape((b, ch, w * h))
        gram = nd.batch_dot(features, features, transpose_b=True)
        return gram

    def get_loss(self, x, y):
        return gluon.loss.L2Loss()(x, y)

    def get_style_loss(self, features, _features):
        loss = 0.
        for i in range(len(features)):
            G = self.get_gram(_features[i])
            A = self.get_gram(features[i])
            N = features[i].shape[1]
            M = features[i] / N
            loss = loss + self.get_loss(A, G) * (1. / (2 * (N ** 2) * (M ** 2))) * 0.2
        return loss
