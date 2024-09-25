import torch
from model.framework.base import Net

class MultiScaleCNN(Net):
    def __init__(self, model_cfg) -> None:
        super().__init__(model_cfg);