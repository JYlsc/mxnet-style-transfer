#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/28 17:09
# @Author  : Leishichi
# @File    : loss.py
# @Software: PyCharm
# @Tag:
from mxnet import gluon









def style_loss(features, _features):
    loss = 0.
    for i in range(len(features)):
        feature = features[i]
        G = gram(feature)
        A = gram(_features[i])
        N = feature.shape[1]
        M = feature.shape[2] * feature.shape[3]
        loss = loss + gluon.loss.L2Loss()(A, G) * (1. / (2 * (N ** 2) * (M ** 2))) * 0.2
    return loss

# def loss(content, _content, style, _style, alpha):
#     loss = 1 * content_loss(content, _content) + alpha * style_loss(style, _style)
#     return loss
