import torch
from lib import util, glb_var
from model import net_util
from model.framework.base import Net
from model import attnet


logger = glb_var.get_value('logger');

class Trans(torch.nn.Module):
    def __init__(self, trans_cfg):
        super().__init__();
        util.set_attr(self, trans_cfg);
        self.embed = attnet.LearnablePositionEncoding(d = self.d, max_len = self.max_len);
        if self.is_norm_first:
            encoderlayer = attnet.EncoderLayer_PreLN;
        else:
            encoderlayer = attnet.EncoderLayer_PostLN;
        self.Layers = torch.nn.ModuleList(
            [encoderlayer(self.d, self.d_fc, self.n_heads) for _ in range(self.n_layers)]
        )
    
    def forward(self, input):
        #input:[N, t, d]
        emb = input + self.embed(input);
        for layer in self.Layers:
            emb = layer(emb);
        return emb;

class AAconv(torch.nn.Module):
    def __init__(self, aaconv_cfg):
        super().__init__();
        util.set_attr(self, aaconv_cfg);
        activation_fn = net_util.get_activation_fn(aaconv_cfg['activation_fn']);
        kernels = aaconv_cfg['hid_layers'];
        layers = [
            net_util.AugmentedConv(
                in_channels = self.in_channels, 
                out_channels = kernels[0], 
                kernel_size = 3, dk = 16, dv = 4, Nh = 4, relative = True, stride = 1, shape = self.shape),
                activation_fn
        ] + [
            net_util.AugmentedConv(
                in_channels = kernels[i], 
                out_channels = kernels[i + 1], 
                kernel_size = 3, dk = 16, dv = 4, Nh = 4, relative = True, stride = 1, shape = self.shape) 
            for i in range(len(kernels) - 1)
        ] + [
            activation_fn,
            net_util.AugmentedConv(
                in_channels = kernels[-1], 
                out_channels = 6, 
                kernel_size = 3, dk = 16, dv = 4, Nh = 4, relative = True, stride = 1, shape = self.shape),
        ];
        self.net = torch.nn.Sequential(*layers);
        self.channel_net = torch.nn.Linear(6, 1);

    def forward(self, input):
        #input:[N, 3， t, d]
        #[N, 6, t, d] -> [N, t, d, 6] -> [N, t, d, 1] -> [N, t, d]
        return self.channel_net(self.net(input).permute(0, 2, 3, 1)).squeeze(-1);

class AAconvTrans(Net):
    def __init__(self, model_cfg):
        super().__init__(model_cfg);
        assert self.t == self.d;
        self.lambda_ = model_cfg['lambda_'];
        #[N, T, R, F] -> [N, T, R, d]
        self.Flinear = torch.nn.Linear(self.dim_in[-1], self.d);

        #intput:[N, d, R, T] -> [N, R, d, t]
        TLinear_cfg = model_cfg['TLinear'];
        TLinear_cfg['activation_fn'] = net_util.get_activation_fn(TLinear_cfg['activation_fn']);
        TLinear_cfg['dim_out'] = self.t;
        TLinear_cfg['dim_in'] = self.dim_in[1];
        self.Tlinear = util.get_func_rets(net_util.get_mlp_net, TLinear_cfg);

        #divided by antenna
        #trans
        trans1_cfg = model_cfg['trans1'];
        trans1_cfg['d'] = self.d;
        self.trans1s = torch.nn.ModuleList([
            Trans(trans1_cfg) for _ in range(self.dim_in[2])
        ])

        aaconv_cfg = model_cfg['aaconv'];
        aaconv_cfg['in_channels'] = 3;
        aaconv_cfg['out_channels'] = 1;
        aaconv_cfg['shape'] = self.t;
        self.aaconv = AAconv(aaconv_cfg);
        
        #tail_net
        tail_net_cfg = model_cfg['tail_net'];
        tail_net_cfg['dim_in'] = self.t * self.d;
        tail_net_cfg['dim_out'] = self.do;
        tail_net_cfg['activation_fn'] = net_util.get_activation_fn(tail_net_cfg['activation_fn']);
        self.tail_net = util.get_func_rets(net_util.get_mlp_net, tail_net_cfg);
    
        #p classifier
        p_c_cfg = model_cfg['p_classifier'];
        p_c_cfg['activation_fn'] = net_util.get_activation_fn(p_c_cfg['activation_fn']);
        p_c_cfg['dim_in'] = self.do;
        p_c_cfg['dim_out'] = self.known_p_num;
        self.p_classifier = util.get_func_rets(net_util.get_mlp_net, p_c_cfg);
    
        #env classifier
        env_c_cfg = model_cfg['env_classifier'];
        env_c_cfg['activation_fn'] = net_util.get_activation_fn(env_c_cfg['activation_fn']);
        env_c_cfg['dim_in'] = self.do;
        env_c_cfg['dim_out'] = self.known_env_num;
        self.env_classifier = util.get_func_rets(net_util.get_mlp_net, env_c_cfg);
    
        self.is_Intrusion_Detection = False;

    def encoder(self, amps):
        #amps:[N, T, R, F] -> [N, T, R, d] -> [N, d, R, T]
        amps1 = self.Flinear(amps).permute(0, 3, 2, 1);
        #amps2:[N, d, R, t] -> [N, t, R, d]
        amps2 = self.Tlinear(amps1).permute(0, 3, 2, 1);

        #([N, 1, t, d], [N, 1, t, d], [N, 1, t, d])
        te1 = tuple([self.trans1s[i](amps2[:, :, i, :]).unsqueeze(1) for i in range(self.dim_in[2])]);
        #[N, 3, t, d]
        te1 = torch.cat(te1, dim = 1);
        #[N, t, d]
        aa = self.aaconv(te1);
        #[N, 256]
        feature = self.tail_net(aa.reshape(amps.shape[0], -1));
        return feature;

    def p_classify(self, amps):
        return self.p_classifier(self.encoder(amps));

    def cal_loss(self, amps, ids, envs, is_target_data = False):
        #amps:[N, T, R, F]
        #ids:[N]
        #envs:[N]
        feature = self.encoder(amps);
        id_probs = self.p_classifier(feature);
        p_loss = torch.nn.CrossEntropyLoss()(id_probs, ids);
        if not is_target_data:
            env_probs = self.env_classifier(net_util.GradientReversalF.apply(feature, self.lambda_));
            env_loss = torch.nn.CrossEntropyLoss()(env_probs, envs);
            return p_loss + env_loss;
        return p_loss;

