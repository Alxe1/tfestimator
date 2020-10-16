# -*-coding: utf-8-*-

# Author     : Littlely 
# FileName   : predictor.py
# Description:
from pathlib import Path

import tensorflow as tf

from iris_estimator.model import model_fn, input_fn
from iris_estimator.utils import from_saved_model


def raw_predict():
    cfg = tf.estimator.RunConfig(save_checkpoints_secs=2)
    estimator = tf.estimator.Estimator(model_fn, model_dir="./my_model", config=cfg, params={})
    result = estimator.predict(input_fn)  # 返回生成器

    return result


def predict(fea):
    export_dir = "./saved_model"
    subdirs = [x for x in Path(export_dir).iterdir()
               if x.is_dir() and 'temp' not in str(x)]
    latest = str(sorted(subdirs)[-1])
    print("加载的模型为：{}".format(latest))

    predict_fn = from_saved_model(latest)

    preds = predict_fn({"input": fea})
    print("预测结果为：{}".format(preds))

    return preds
