import torch
from lib import util, glb_var
import numpy as np
from model import net_util, attnet
from model.framework.base import Net

class CNNBlock(torch.nn.Module):
    def __init__(self, channel_in, channel_out, avgpool) -> None:
        super().__init__();
        net = [
            torch.nn.Conv1d(channel_in, channel_out, kernel_size = 5, padding = 2),
            torch.nn.BatchNorm1d(channel_out),
            torch.nn.MaxPool1d(kernel_size = 2, stride = 2)
        ];
        if avgpool:
            net = net + [torch.nn.AvgPool1d(kernel_size = 2, stride = 2)];
        self.net = torch.nn.Sequential(*net);

    def forward(self, x):
        return self.net(x);

class MultiScaleCNN(Net):
    def __init__(self, model_cfg) -> None:
        super().__init__(model_cfg);
        channels = [self.channel_in] + self.h_layers;
        # H net
        self.hnet = torch.nn.ModuleList([
            torch.nn.Sequential(*[
                torch.nn.Sequential(
                    torch.nn.Conv1d(channels[i], channels[i + 1], kernel_size = 1, padding = 0),
                    torch.nn.ReLU()
                )
                for i in range(len(channels) - 1)
            ]),
            torch.nn.Sequential(*[
                torch.nn.Sequential(
                    torch.nn.Conv1d(channels[i], channels[i + 1], kernel_size = 5, padding = 2),
                    torch.nn.ReLU()
                )
                for i in range(len(channels) - 1)
            ]),
            torch.nn.Sequential(*[
                torch.nn.Sequential(
                    torch.nn.Conv1d(channels[i], channels[i + 1], kernel_size = 7, padding = 3),
                    torch.nn.ReLU()
                )
                for i in range(len(channels) - 1)
            ])
        ]);

        # H linear
        hlinear_cfg = {
            'hid_layers':[],
            'activation_fn':None,
            'drop_out':None,
            'dim_in':self.dim_in[1],
            'dim_out':1,
            'end_with_softmax':True
        }
        self.hlinears = torch.nn.ModuleList([util.get_func_rets(net_util.get_mlp_net, hlinear_cfg) for _ in range(3)]);

        #conv net 
        channels = [channels[-1]] + self.hid_layers;
        self.convnet = torch.nn.Sequential(*[
            CNNBlock(channels[i], channels[i + 1], avgpool = (i + 2 == len(channels)))
            for i in range(len(channels) - 1)
        ]);

        #feature linear
        flinear_cfg = {
            'hid_layers':[],
            'activation_fn':None,
            'drop_out':None,
            'dim_in': (self.dim_in[1] // 2 ** len(channels)) * channels[-1],
            'dim_out':self.do,
            'end_with_softmax':True
        };
        self.feature_linear = util.get_func_rets(net_util.get_mlp_net, flinear_cfg);

    def forward(self, x):
        #x:[B, T, R * F]
        x = x.permute(0, 2, 1);
        Hs = [self.hnet[i](x) for i in range(3)];
        alphas = [self.hlinears[i](Hs[i]) for i in range(3)];
        z = sum([alphas[i] * Hs[i] for i in range(3)]);
        return self.feature_linear(self.convnet(z).flatten(1, -1));

class WiAiId(Net):
    def __init__(self, model_cfg) -> None:
        super().__init__(model_cfg);
        d = np.prod(self.dim_in[2:]);
        #Position encoding
        self.pos_enc = attnet.LearnablePositionEncoding(d = d, max_len = self.dim_in[1]);
        # MHA
        MHA_cfg = model_cfg['MHA_cfg'];
        MHA_cfg['d'] = d;
        self.MHA = attnet.MultiHeadAttention(**MHA_cfg);

        #MultiScaleCNN
        MultiScaleCNN_cfg = model_cfg['MultiScaleCNN_cfg'];
        MultiScaleCNN_cfg['do'] = self.do;
        MultiScaleCNN_cfg['channel_in'] = np.prod(self.dim_in[2:]);
        MultiScaleCNN_cfg['dim_in'] = self.dim_in;
        self.MultiScaleCNN = MultiScaleCNN(MultiScaleCNN_cfg);

        # p_classifier
        layers = [self.do] + model_cfg['p_classifier']['hid_layers'] + [self.known_p_num];
        self.p_classifier = torch.nn.Sequential(*[
            torch.nn.Sequential(
                torch.nn.Linear(layers[i], layers[i + 1]),
                torch.nn.ReLU()
            )
            for i in range(len(layers) - 1)
        ] + [
            torch.nn.Softmax(dim = -1)
        ]);

        # p net
        self.p_net = torch.nn.Sequential(
            torch.nn.Linear(self.known_p_num, self.do),
            torch.nn.Dropout()
        );

        # env_classifier
        layers = [self.do] + model_cfg['env_classifier']['hid_layers'] + [self.known_env_num + 1];
        self.env_classifier = torch.nn.Sequential(*[
            torch.nn.Sequential(
                torch.nn.Linear(layers[i], layers[i + 1]),
                torch.nn.ReLU()
            )
            for i in range(len(layers) - 1)
        ] + [
            torch.nn.Softmax(dim = -1)
        ]);

        self.is_Intrusion_Detection = False;

    def encoder(self, amps):
        #[B, T, R, F]
        amps = amps.flatten(2, -1);
        pos_embs = amps + self.pos_enc(amps);
        features = self.MHA(pos_embs, pos_embs, pos_embs);
        features = self.MultiScaleCNN(features);
        #[B, do]
        return features;

    def p_classify(self, amps):
        return self.p_classifier(self.encoder(amps));

    def cal_loss(self, amps, ids, envs, amps_t = None, ids_t = None, envs_t = None):
        #[B, do]
        feature_s = self.encoder(amps);
        #Identification of source
        id_s_probs = self.p_classifier(feature_s);
        loss_id = torch.nn.CrossEntropyLoss()(id_s_probs, ids);
        #env loss of  source
        refeture_s = self.p_net(id_s_probs);
        env_s_probs = self.env_classifier(refeture_s + feature_s);
        loss_env_s = torch.nn.CrossEntropyLoss()(env_s_probs, envs);
        if amps_t is None:
            #valid
            return loss_id +  loss_env_s;
        #[B, do]
        feature_t = self.encoder(amps_t);

        #loss of feature between source and target
        loss_mmd = net_util.mmd_loss(feature_s, feature_t);
        loss_coral = net_util.coral_loss(feature_s, feature_t);

        #env loss of  target
        refeture_t = self.p_net(self.p_classifier(feature_t));
        env_t_probs = self.env_classifier(refeture_t + feature_t);
        loss_env_t = torch.nn.CrossEntropyLoss()(env_t_probs, envs_t);
        loss_env = loss_env_s + loss_env_t;
        return loss_id - self.alpha * loss_env + self.beta * loss_mmd + self.gamma * loss_coral;

glb_var.register_model('WiAiId', WiAiId);