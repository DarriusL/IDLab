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
    
    @torch.no_grad()
    def encoder(self, amps):
        #amps:[B, T, R * F]
        amps = amps.reshape(amps.shape[0], amps.shape[1], -1);
        #[B, T, d]
        features, _ = self.gru(amps);
        #[B, d]
        features = features.mean(dim = 1);
        return features;

    def p_classify(self, amps):
        #amps:[B, T, R * F]
        #[B, d]
        features = self.encoder(amps);
        id_probs = self.tail_net(features);
        return id_probs;

    def cal_loss(self, amps, ids, envs):
        id_probs = self.p_classify(amps);
        return torch.nn.CrossEntropyLoss()(id_probs, ids);

class DeepWiIDAL(DeepWiID):
    def __init__(self, model_cfg):
        super().__init__(model_cfg);

        #env layer
        env_cfg = model_cfg['env_classifier'];
        env_cfg['activation_fn'] = net_util.get_activation_fn(env_cfg['activation_fn']);
        env_cfg['dim_in'] = model_cfg['tail_net']['dim_in'];
        env_cfg['dim_out'] = self.known_env_num;
        self.env_layer = util.get_func_rets(net_util.get_mlp_net, env_cfg);

    def cal_loss(self, amps, ids, envs, amps_t = None, ids_t = None, envs_t = None):
        features = self.encoder(amps);
        id_probs = self.tail_net(features);
        loss_id = torch.nn.CrossEntropyLoss()(id_probs, ids);

        if amps_t is None:
            return loss_id;

        env_probs = self.env_layer(net_util.GradientReversalF.apply(features, self.lambda_));
        loss_env = torch.nn.CrossEntropyLoss()(env_probs, envs);

        loss_t = loss_id + loss_env;

        feature_t = self.encoder(amps_t);
        id_probs_t = self.tail_net(feature_t);
        id_loss_t = torch.nn.CrossEntropyLoss()(id_probs_t, ids_t);

        return loss_t + id_loss_t;

glb_var.register_model('DeepWiID', DeepWiID);
glb_var.register_model('DeepWiIDAL', DeepWiIDAL);