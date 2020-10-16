# -*-coding: utf-8-*-

# Author     : Littlely 
# FileName   : train_and_save_model.py
# Description:

from iris_estimator.model import build_estimator, model_fn
from iris_estimator.export_model import export_model


if __name__ == '__main__':
    build_estimator()
    export_model(model_fn, "./my_model", None, None, "saved_model")
