#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/25 17:04
# @Author  : Leishichi
# @File    : model.py
# @Software: PyCharm
# @Tag:

import mxnet as mx
from mxnet import gluon


class FeatureNet():

    def __init__(self, vgg):
        self.net = self.init_net(vgg)

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


def features_net(vgg):
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
