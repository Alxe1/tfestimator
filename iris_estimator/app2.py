# -*-coding: utf-8-*-

# Author     : Littlely 
# FileName   : app2.py
# Description:
import json
import requests


def serving(data):
    """
    用于tensorflow serving
    :param data: numpy array or list
    :return:
    """
    data = {"instances": data}
    data = json.dumps(data)
    # response = requests.post("http://localhost:5000/serving", data=data)  # 可用于app1.serving2
    response = requests.post("http://localhost:8501/v1/models/saved_model:predict", data=data)
    x = response.content.decode("utf-8")
    print("状态码为：{}".format(response.status_code))
    print("预测结果为：{}".format(json.loads(x)))


if __name__ == '__main__':
    d = [[5.0, 2.0, 3.5, 1.0], [6.3, 2.5, 5.0, 1.9]]
    serving(d)
