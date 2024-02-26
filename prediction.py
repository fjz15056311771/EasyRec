#!/usr/bin/env python

from eas_prediction import PredictClient
from eas_prediction import StringRequest
from eas_prediction.blade_tf_request import TFRequest
import tensorflow as tf


if __name__ == '__main__':
    client = PredictClient('http://1005963784283285.cn-hangzhou.pai-eas.aliyuncs.com/api/predict/deepfm2')
    client.set_token('ZGY2YTE4MjQ1MzkzZGMyZGFlZGE5OTE0NGJmYzMyZWQ4MDdlODE2Yw==')
    client.init()

    req = TFRequest('prediction')
    req.add_feed('features', [1], TFRequest.DT_STRING, tf.train.Feature(bytes_list=tf.train.BytesList(value=[4819])))
    for x in range(0, 1000000):
        resp = client.predict(req)
        print(resp)
    # request = StringRequest('[{"fea1": 1, "fea2": 2}]')
    # for x in range(0, 1000000):
    #     resp = client.predict(request)
    #     print(resp)