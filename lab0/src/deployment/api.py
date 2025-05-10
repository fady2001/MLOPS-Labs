import os
import pickle

import litserve as ls
import numpy as np
import pandas as pd

from src.deployment.requests import InferenceRequest


class InferenceAPI(ls.LitAPI):
    def __init__(self, cfg):
        self.cfg = cfg
        self.max_batch_size = 1  # Set the maximum batch size (adjust as needed)
        self.enable_async = False  # Set to True if you want to enable async processing
        self.batch_timeout = 0.1  # Set the batch timeout (in seconds)
        print(cfg)

    def setup(self, device="cpu"):
        with open(
            os.path.join(
                self.cfg["paths"]["models_parent_dir"],
                self.cfg["names"]["model_name"],
                f"{self.cfg['names']['model_name']}.pkl",
            ),
            "rb",
        ) as pkl:
            self._model = pickle.load(pkl)

    def decode_request(self, request):
        try:
            # InferenceRequest(**request["dataframe_split"])
            columns = request["dataframe_split"]["columns"]
            data = request["dataframe_split"]["data"]

            df = pd.DataFrame(data, columns=columns)
            return df
        except Exception:
            return None

    def predict(self, x):
        print(x)
        if x is not None:
            return self._model.predict(x)
        else:
            return None

    def encode_response(self, output):
        print(output, 9 * "*")
        if output is None:
            message = "Error Occurred"
        else:
            message = "Response Produced Successfully"
        response = {
            "message": message,
            "data": output.tolist(),
        }
        return response
