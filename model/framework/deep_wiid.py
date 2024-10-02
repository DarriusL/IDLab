import torch
import numpy as np
from lib import util, glb_var
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
        #There is only one layer of network, and these two parameters are not needed
        tail_net_cfg['activation_fn'] = net_util.get_activation_fn(tail_net_cfg['activation_fn']) if 'activation_fn' in tail_net_cfg.keys() else None;
        tail_net_cfg['drop_out'] = tail_net_cfg['drop_out'] if 'drop_out' in tail_net_cfg.keys() else None;
        self.tail_net = util.get_func_rets(net_util.get_mlp_net, tail_net_cfg);

        self.is_Intrusion_Detection = False;

    def p_classify(self, amps):
        #amps:[B, T, R * F]
        amps = amps.reshape(amps.shape[0], amps.shape[1], -1);
        #[B, T, d]
        features, _ = self.gru(amps);
        #[B, d]
        features = features.mean(dim = 1);
        id_probs = self.tail_net(features);
        return id_probs;

    def cal_loss(self, amps, ids, envs):
        id_probs = self.p_classify(amps);
        return torch.nn.CrossEntropyLoss()(id_probs, ids);

glb_var.register_model('DeepWiID', DeepWiID);