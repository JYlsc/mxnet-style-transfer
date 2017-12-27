#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/12/25 16:58
# @Author  : Leishichi
# @File    : net_tool.py
# @Software: PyCharm
# @Tag:

import numpy as np
import mxnet as mx
from mxnet import image
import mxnet.ndarray as nd
import cv2


def read_img(path, size=300):
    """
    读取图片
    :param path: 图片路径
    :return: shape = (1,3,224,224) , layout =（B,C,H,D）
    """
    # 读取图片
    img = nd.array(cv2.imread(path))
    # 转换图片大小
    img = image.imresize(img,size,size)
    # 将图片归一化
    img = img.astype('float32')/255
    # 将图片由 (H,W,C）转换成 (N,C,H,W)
    img = nd.transpose(img, axes=(2, 0, 1)).expand_dims(0)
    return img

def new_img(shape):
    return mx.nd.random.normal(shape=shape)


def save_img(img, path):
    """
    储存一张图片
    :param img: layout =（B,C,H,D）
    :param path: 储存路径
    :return:
    """
    # 将图片由 (N,C,H,W)转换成 (C,H,W)
    img = img[0,]
    # 将图片由 (C,H,W)转换成 (H,W,C)
    img = nd.transpose(img, axes=(1, 2, 0))*255

    img = img.asnumpy()
    cv2.imwrite(path,img)

    # plt.imshow(img.asnumpy().astype(np.uint8))
    # plt.savefig(path)
    # plt.show()


def get_ctx():
    """
    判断是否能使用GPU
    :return:
    """
    try:
        ctx = mx.gpu()
        _ = nd.zeros((1,), ctx=ctx)
    except:
        ctx = mx.cpu()
    return ctx
