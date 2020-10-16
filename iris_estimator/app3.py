# -*-coding: utf-8-*-

# Author     : Littlely 
# FileName   : app3.py
# Description:
import grpc
from tensorflow_serving.apis import prediction_service_pb2_grpc, predict_pb2
from tensorflow.python.platform import flags
from tensorflow.python.framework import tensor_util


FLAGS = flags.FLAGS
flags.DEFINE_string("server", "localhost:8500", "PredictionService host:port")


def serving(input_data):
    """
    用于tensorflow serving
    :param input_data: 输入数据，numpy array or list
    :return:
    """
    channel = grpc.insecure_channel(FLAGS.server)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = "saved_model"  # 保存模型的名称
    request.model_spec.signature_name = "serving_default"

    tensor_proto = tensor_util.make_tensor_proto(input_data)
    request.inputs["input"].CopyFrom(tensor_proto)  # `input`为模型输入的key

    # result = stub.Predict(request, 5.0)
    future = stub.Predict.future(request, 5)  # 并发运行
    result = future.result()
    print(result)
    res = result.outputs["pred"].int64_val
    prob = result.outputs["prob"].float_val
    print("预测结果：{}".format(list(res)))
    print("预测概率：{}".format(list(prob)))


if __name__ == '__main__':
    input_data = [[5.0, 2.0, 3.5, 1.0], [6.3, 2.5, 5.0, 1.9]]
    serving(input_data)
