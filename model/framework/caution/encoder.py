import torch
from lib import util, glb_var
from model import net_util
from model.framework.base import Net

device = glb_var.get_value('device');

class Encoder(Net):
    def __init__(self, model_cfg) -> None:
        super().__init__(model_cfg);

        #input:[B, 6000, 3, 56] -> [B, 3, 6000, 56] -> [B, 128, 750, 7]
        #128 = channels[-1]
        channel_in = self.dim_in[2];
        channels = model_cfg['cnn']['hid_layer'];
        assert len(channels) == 3
        activation_fn = net_util.get_activation_fn(model_cfg['cnn']['activation_fn']);
        layers = [
            torch.nn.Conv2d(channel_in, channels[0], kernel_size = (5, 5), padding = 2),
            activation_fn,
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        ] + [
            layer
            for i in range(len(channels) - 1)
            for layer in [
            torch.nn.Conv2d(channels[i], channels[i + 1], kernel_size=(5, 5), padding=2),
            activation_fn,
            torch.nn.MaxPool2d(kernel_size=2, stride=2)]
        ];
        self.cnn = torch.nn.Sequential(*layers);

        #[B, 128, 750, 7] -> [B, do]
        FL_cfg = model_cfg['FeatureLayer'];
        FL_cfg['activation_fn'] = net_util.get_activation_fn(FL_cfg['activation_fn']);
        FL_cfg['dim_out'] = self.do;
        FL_cfg['dim_in'] = channels[-1] * 750 * 7;
        self.FeatureLayer = util.get_func_rets(net_util.get_mlp_net, FL_cfg);
        self.pcenters = None;
        self.is_Intrusion_Detection = False
    
    def encoder(self, amps):
        #amps:[B, 6000, 3, 56] -> [B, 3, 6000, 56]
        feature1 = self.cnn(amps.permute(0, 2, 1, 3));
        feature2 = self.FeatureLayer(feature1.reshape(amps.shape[0], -1));
        return feature2;

    @torch.no_grad()
    def update_center(self, support_loader):
        self.eval();
        self.pcenters = torch.zeros((self.known_p_num, self.do), device = device);
        p_id = 0;
        for amps, ids, _ in iter(support_loader):
            #amps:[1, B, 6000, 3, 56]
            assert torch.all(ids == p_id);
            #[B, 6000, 3, 56] -> [B, do] -> [do]
            self.pcenters[p_id, :] = self.encoder(amps.squeeze(0)).mean(dim = 0); 
            p_id += 1;
        self.train();
    
    def p_classify(self, amps):
        #[B, do]
        features = self.encoder(amps);
        #[B, N]
        distences = torch.exp( - torch.cdist(features, self.pcenters));
        #[B]
        probs = distences / distences.sum(dim = 1, keepdim = True);
        return probs;

    def cal_loss(self, amps, ids, envs, is_target_data = False):
        probs = self.p_classify(amps);
        return torch.nn.CrossEntropyLoss()(probs, ids);

    