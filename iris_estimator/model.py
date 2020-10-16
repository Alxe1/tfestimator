# -*-coding: utf-8-*-

# Author     : Littlely 
# FileName   : model.py
# Description:
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.logging.INFO)


def data_generator():
    with open("./data/iris.data", 'r', encoding="utf-8") as f:
        for line in f:
            arr = line.strip().split(",")
            yield [arr[:-1]], 1 if arr[-1] == "Iris-setosa" else 2 if arr[-1] == "Iris-versicolor" else 0


def input_fn():
    output_shape = ([None, None], ())
    output_type = (tf.float32, tf.int32)
    dataset = tf.data.Dataset.from_generator(data_generator, output_type, output_shape)
    dataset = dataset.repeat(50).shuffle(50).batch(20)
    dataset = dataset.map(map_func=lambda x, y: (x, tf.one_hot(y, 3)))

    return dataset


def model_fn(features, labels, mode, params):
    if isinstance(features, dict):
        input = features["input"]
    else:
        input = features
    input = tf.reshape(input, [-1, 4])

    dense_1 = tf.layers.dense(input, 10, activation=tf.nn.relu, kernel_initializer=tf.initializers.random_normal(), name="dense_1")
    dense_2 = tf.layers.dense(dense_1, 10, activation=tf.nn.relu, kernel_initializer=tf.initializers.random_normal(), name="dense_2")

    w = tf.get_variable("weight", shape=[10, 3], dtype=tf.float32, initializer=tf.variance_scaling_initializer())

    logits = tf.matmul(dense_2, w)

    prob = tf.nn.softmax(logits)
    prob = tf.reduce_max(prob, axis=1)

    if mode == tf.estimator.ModeKeys.TRAIN:
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels, logits)
        loss = tf.reduce_mean(loss)

        logging_hook = tf.estimator.LoggingTensorHook({"loss": loss, "prob": prob}, every_n_iter=1)

        optimizer = tf.train.GradientDescentOptimizer(0.01)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode,  loss=loss, train_op=train_op, training_hooks=[logging_hook])

    if mode == tf.estimator.ModeKeys.EVAL:
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels, logits)
        loss = tf.reduce_mean(loss)
        return tf.estimator.EstimatorSpec(mode, loss=loss)

    if mode == tf.estimator.ModeKeys.PREDICT:
        pred = tf.argmax(logits, axis=-1)
        predictions = {
            "pred": pred,
            "prob": prob
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)


def build_estimator():
    print("================开始训练===================")
    cfg = tf.estimator.RunConfig(save_checkpoints_secs=2)
    estimator = tf.estimator.Estimator(model_fn, model_dir="./my_model", config=cfg, params={})

    train_spec = tf.estimator.TrainSpec(input_fn)
    eval_spec = tf.estimator.EvalSpec(input_fn=input_fn)

    v = tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    print("================完成训练====================")
