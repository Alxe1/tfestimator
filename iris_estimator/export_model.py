# -*-coding: utf-8-*-

# Author     : Littlely 
# FileName   : export_model.py
# Description:
import tensorflow as tf


def serving_input_receiver_fn():
    input = tf.placeholder(dtype=tf.float32, shape=[None, 4], name="input")
    features = {"input": input}
    receiver_tensors = {"input": input}

    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


def export_model(model_fn, ckp_dir, config, params, save_path):
    estimator = tf.estimator.Estimator(model_fn, ckp_dir, config=config, params=params)
    estimator.export_saved_model(save_path, serving_input_receiver_fn)
    print("保存模型至：{}".format(save_path))

