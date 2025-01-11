import torch, scipy
from model.framework.base import Net
from lib import util, glb_var
from model import net_util
import numpy as np

device = glb_var.get_value('device');

class CnnBlock(torch.nn.Module):
    def __init__(self, blk_cfg) -> None:
        super().__init__();
        conv = torch.nn.Conv2d(
            in_channels = blk_cfg['channel_in'], 
            out_channels = blk_cfg['channel_out'], 
            kernel_size = 3, 
            padding=1);
        bn = torch.nn.BatchNorm2d(blk_cfg['channel_out']);
        relu = torch.nn.ReLU();
        pool = torch.nn.MaxPool2d(kernel_size=2, stride=2);
        self.net = torch.nn.Sequential(*[conv, bn, relu, pool]);
        

    def forward(self, amps):
        #amps:[B, C, , ,]
        return self.net(amps);

class GaitEnhance(Net):
    def __init__(self, model_cfg) -> None:
        super().__init__(model_cfg);
        blks_cfg = model_cfg['blks_cfg'];
        assert len(blks_cfg['hid_layers']) == 3
        channels = [self.dim_in[2], ] + blks_cfg['hid_layers'];
        blk_cfgs = [ {
                        'channel_in':channels[i],
                        'channel_out':channels[i + 1],
                    }
                    for i in range(len(blks_cfg['hid_layers']))];
        self.convnet = torch.nn.Sequential(*[
            CnnBlock(blk_cfg) for blk_cfg in blk_cfgs
        ]);
        self.dropout = torch.nn.Dropout(self.dropout_rate);

        #output layer
        o_cfg = model_cfg['output_layer'];
        o_cfg['activation_fn'] = net_util.get_activation_fn(o_cfg['activation_fn']);
        f_dim = self._cal_features_dim();
        o_cfg['dim_in'] = channels[-1] * (f_dim[1] // 2 ** len(blks_cfg['hid_layers'])) * (f_dim[-1] // 2 ** len(blks_cfg['hid_layers']));
        o_cfg['dim_out'] = self.known_p_num;
        self.out_layer = util.get_func_rets(net_util.get_mlp_net, o_cfg);
    
    def _cal_features_dim(self):
        return self._ext_features(torch.rand(self.dim_in, device = device)).shape;

    def _ext_features(self, amps):
        #amps:[B, T, R, F]
        #amps_window:[B, T/w, R, F, w]
        amps_window = amps.unfold(1, self.window_size, self.window_size);
        #[B, T/w, R, F]
        amps_max = amps_window.max(dim = -1)[0];
        amps_min = amps_window.min(dim = -1)[0];
        amps_mean = amps_window.mean(dim = -1);
        amps_std = amps_window.std(dim = -1);

        amps_window_np = amps_window.cpu().numpy();
        #Normlize to avoid
        # RuntimeWarning: Precision loss occurred in moment calculation due to catastrophic cancellation. 
        # This occurs when the data are nearly identical. Results may be unreliable.
        #amps_skew = torch.tensor(scipy.stats.skew(amps_window_np, axis=-1), device = device);
        amps_window_np_normalized = (amps_window_np - np.mean(amps_window_np, axis=-1, keepdims=True)) / np.std(amps_window_np, axis=-1, keepdims=True);
        amps_skew = torch.tensor(scipy.stats.skew(amps_window_np_normalized, axis=-1), device=device);

        #amps_skew = torch.tensor(scipy.stats.skew(amps_window_np, axis=-1), device = device);
        amps_kurtosis = torch.tensor(scipy.stats.kurtosis(amps_window_np_normalized, axis=-1), device = device);

        median_vals = torch.median(amps_window, dim=-1)[0];
        median_deviation_vals = torch.median(torch.abs(amps_window - median_vals.unsqueeze(-1)), dim=-1)[0];

        return torch.cat([
            amps_max, amps_min, amps_mean,
            amps_std, amps_skew, amps_kurtosis,
            median_deviation_vals
        ], dim = 1);

    @torch.no_grad()
    def encoder(self, amps):
        features = self._ext_features(amps).permute(0, 2, 1, 3);
        return self.dropout(self.convnet(features));

    def p_classify(self, amps):
        features = self.encoder(amps);
        id_probs = self.out_layer(features.flatten(1, -1));
        return id_probs;

    def cal_loss(self, amps, ids, envs):
        id_probs = self.p_classify(amps);
        return torch.nn.CrossEntropyLoss()(id_probs, ids);

class GaitEnhanceAL(GaitEnhance):
    def __init__(self, model_cfg):
        super().__init__(model_cfg);
        self.lambda_ = model_cfg['lambda_'];

        #env layer
        env_cfg = model_cfg['env_classifier'];
        env_cfg['activation_fn'] = net_util.get_activation_fn(env_cfg['activation_fn']);
        env_cfg['dim_in'] = model_cfg['output_layer']['dim_in'];
        env_cfg['dim_out'] = self.known_env_num;
        self.env_layer = util.get_func_rets(net_util.get_mlp_net, env_cfg);

    def cal_loss(self, amps, ids, envs, amps_t = None, ids_t = None, envs_t = None):
        features = self.encoder(amps);
        id_probs = self.out_layer(features.flatten(1, -1));
        loss_id = torch.nn.CrossEntropyLoss()(id_probs, ids);

        if amps_t is None:
            return loss_id;

        env_probs = self.env_layer(net_util.GradientReversalF.apply(features.flatten(1, -1), self.lambda_));
        loss_env = torch.nn.CrossEntropyLoss()(env_probs, envs);

        loss_t = loss_id + loss_env;

        feature_t = self.encoder(amps_t);
        id_probs_t = self.out_layer(feature_t.flatten(1, -1));
        id_loss_t = torch.nn.CrossEntropyLoss()(id_probs_t, ids_t);

        return loss_t + id_loss_t;



glb_var.register_model('GaitEnhance', GaitEnhance);
glb_var.register_model('GaitEnhanceAL', GaitEnhanceAL);