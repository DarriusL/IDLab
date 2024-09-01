import torch
from lib import util, glb_var
from model import net_util
from model.framework.base import Net

class Caution(Net):
    def __init__(self, model_cfg) -> None:
        super().__init__(model_cfg);
        