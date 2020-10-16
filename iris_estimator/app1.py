# -*-coding: utf-8-*-

# Author     : Littlely 
# FileName   : app1.py
# Description:
import json

from flask import Flask, jsonify, request
from iris_estimator.predictor import predict

app = Flask(__name__)


@app.route('/serving/<data>')
def serving(data):
    """
    get请求
    :param data:
    :return:
    """
    feature = eval(data)
    preds = predict(feature)

    preds = jsonify(str(preds))

    return preds

# @app.route('/serving', methods=["POST"])
# def serving2():
#     """
#     该函数可以使用post请求对数据进行处理
#     :return:
#     """
#     if request.method == "POST":
#         data = request.get_data()
#         json_data = json.loads(data.decode("utf-8"))
#         print(json_data)
#
#         inputs = json_data["instances"]
#         preds = predict(inputs)
#         preds = jsonify(str(preds))
#
#         return preds
#
#     return ValueError("Maybe request method should be POST or the predict method is not serving.")


if __name__ == '__main__':
    app.run()
