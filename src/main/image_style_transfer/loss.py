#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/28 17:09
# @Author  : Leishichi
# @File    : loss.py
# @Software: PyCharm
# @Tag:
from mxnet import gluon
from mxnet import ndarray as nd


def get_gram(features):
    list = []
    for feature in features:
        list.append(gram(feature))
    return list


def gram(features):
    """
    计算features 的 gram矩阵
    :param features:
    :return:
    """
    c = features.shape[1]
    n = features.size / features.shape[1]
    y = features.reshape((c, int(n)))
    return nd.dot(y, y.T) / n


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
