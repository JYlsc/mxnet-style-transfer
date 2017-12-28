#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/28 17:09
# @Author  : Leishichi
# @File    : loss.py
# @Software: PyCharm
# @Tag:
from mxnet import gluon
from mxnet import ndarray as nd


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
    gram = nd.batch_dot(features, features, transpose_b=True)
    return gram


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


def loss(content, _content, style, _style, alpha):
    loss = 1 * content_loss(content, _content) + alpha * style_loss(style, _style)
    return loss
