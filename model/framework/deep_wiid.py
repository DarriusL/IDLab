import torch
import numpy as np
from lib import util
from model.framework.base import Net
from model import net_util

class DeepWiID(Net):
    def __init__(self, model_cfg) -> None:
        super().__init__(model_cfg);
        #GRU
        gru_cfg = model_cfg['gru_cfg'];
        gru_cfg['input_size'] = np.prod(self.dim_in[2:]);
        self.gru = torch.nn.GRU(**gru_cfg);
        #Tail
        #tail_net
        tail_net_cfg = model_cfg['tail_net'];
        tail_net_cfg['dim_in'] = gru_cfg['hidden_size'];
        tail_net_cfg['dim_out'] = self.known_p_num;
        tail_net_cfg['activation_fn'] = net_util.get_activation_fn(tail_net_cfg['activation_fn']);
        self.tail_net = util.get_func_rets(net_util.get_mlp_net, tail_net_cfg);

        self.is_Intrusion_Detection = False;

    def p_classify(self, amps):
        #amps:[batch_size]
        amps = amps.reshape(amps.shape[0], amps.shape[1], -1)
