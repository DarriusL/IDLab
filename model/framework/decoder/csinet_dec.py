import torch
import numpy as np
from lib import util, glb_var
from model import net_util
from model.framework.base import Net

logger = glb_var.get_value('logger');
device = glb_var.get_value('device');

class RefineNet(torch.nn.Module):
    def __init__(self, refinenet_cfg):
        super().__init__();
        activation_fn = net_util.get_activation_fn(refinenet_cfg['activation_fn']);
        channel = refinenet_cfg['in_channels'];
        kernel = refinenet_cfg['hid_layers'];
        layers = [
            torch.nn.Conv2d(channel, kernel[0], kernel_size=3, padding=1),
            activation_fn
        ] + [
            torch.nn.Conv2d(kernel[i], kernel[i + 1], kernel_size=3, padding=1) 
            for i in range(len(kernel) - 1)
        ] + [
            activation_fn,
            torch.nn.Conv2d(kernel[-1], channel, kernel_size=3, padding=1)
        ];
        self.net = torch.nn.Sequential(*layers);

    def forward(self, input):
        #input:[N, t, d, d]
        return input + self.net(input);

class CSINet(torch.nn.Module):
    def __init__(self, refinenet_cfg) -> None:
        super().__init__();
        layers = [
            RefineNet(refinenet_cfg) for _ in range(refinenet_cfg['n_refine'])
        ]
        self.net = torch.nn.Sequential(*layers);

    def forward(self, input):
        #input:[N, t, d, d]
        #output:[N, t, d, d]
        return self.net(input);

class CSINetDec(Net):
    def __init__(self, model_cfg) -> None:
        super().__init__(model_cfg);
        #load and freeze Parameters
        self.pretrained_enc = torch.load(self.pretrained_enc_dir)['model'];
        for param in self.pretrained_enc.parameters():
            param.requires_grad = False;
        
        #Preamble Assertions
        assert self.d ** 2 == self.do;
        assert self.do == self.pretrained_enc.do;
        
        #input:[N, 1, d, d] -> [N, t, d, d]
        TLinear1_cfg = model_cfg['TLinear1'];
        TLinear1_cfg['activation_fn'] = net_util.get_activation_fn(TLinear1_cfg['activation_fn']);
        TLinear1_cfg['dim_out'] = self.t;
        TLinear1_cfg['dim_in'] = 1;
        self.Tlinear1 = util.get_func_rets(net_util.get_mlp_net, TLinear1_cfg);

        #csinet
        csinet_cfg = model_cfg['csinet'];
        csinet_cfg['in_channels'] = self.t;
        csinet_cfg['shape'] = self.d;
        self.csinet = CSINet(csinet_cfg);

        #input:[N, t, d, d] -> [N, t, d, F]
        FLinear_cfg = model_cfg['FLinear'];
        FLinear_cfg['activation_fn'] = net_util.get_activation_fn(FLinear_cfg['activation_fn']);
        FLinear_cfg['dim_out'] = self.dim_in[-1];
        FLinear_cfg['dim_in'] = self.d;
        self.Flinear = util.get_func_rets(net_util.get_mlp_net, FLinear_cfg);

        #input:[N, t, d, F] -> [N, t, R, F]
        RLinear_cfg = model_cfg['RLinear'];
        RLinear_cfg['activation_fn'] = net_util.get_activation_fn(RLinear_cfg['activation_fn']);
        RLinear_cfg['dim_out'] = self.dim_in[2];
        RLinear_cfg['dim_in'] = self.d;
        self.Rlinear = util.get_func_rets(net_util.get_mlp_net, RLinear_cfg);

        #input:[N, t, d, F] -> [N, T, R, F]
        TLinear2_cfg = model_cfg['TLinear2'];
        TLinear2_cfg['activation_fn'] = net_util.get_activation_fn(TLinear2_cfg['activation_fn']);
        TLinear2_cfg['dim_out'] = self.dim_in[1];
        TLinear2_cfg['dim_in'] = self.t;
        self.Tlinear2 = util.get_func_rets(net_util.get_mlp_net, TLinear2_cfg);

        self.loss_func = net_util.get_loss_func(model_cfg['loss']);
        self.threshold = .0;
        self.thresholds = dict();

        self.is_Intrusion_Detection = True;

    def reconstruct(self, amps):
        with torch.no_grad():
            #[N, T, R, F] -> [N, do] -> [N, 1, d, d]
            feature1 = self.pretrained_enc.encoder(amps).reshape(-1, 1, self.d, self.d);
        #[N, 1, d, d] -> [N, d, d, 1] -> [N, d, d, t] -> [N, t, d, d]
        feature2 = self.Tlinear1(feature1.permute(0, 3, 2, 1)).permute(0, 3, 2, 1);
        #[N, t, d, d]
        feature3 = self.csinet(feature2);
        #[N, t, d, d] -> [N, t, d, F]
        feature4 = self.Flinear(feature3);
        #[N, t, d, F] -> [N, t, F, d] -> [N, t, F, R] -> [N, t, R, F]
        feature5 = self.Rlinear(feature4.permute(0, 1, 3, 2)).permute(0, 1, 3, 2);
        #[N, t, R, F] -> [N, R, F, t] -> [N, R, F, T] -> [N, T, R, F]
        csi_rcst = self.Tlinear2(feature5.permute(0, 2, 3, 1)).permute(0, 3, 1, 2);
        return csi_rcst;

    def cal_loss(self, amps, ids, envs, is_target_data = False, keep_batch = False):
        #amps:[N, T, R, F]
        #ids:[N]
        #envs:[N]
        csi_rcst = self.reconstruct(amps);
        loss = self.loss_func(csi_rcst, amps, keep_batch);
        return loss;

    @torch.no_grad()
    def update_thresholds(self, loader):
        self.thresholds = dict();
        loader.disable_aug();
        self.eval();
        rcst_errors = torch.zeros(0, device = device);
        for amps, ids, envs in iter(loader):
            rcst_errors = torch.cat((
                rcst_errors,
                self.cal_loss(amps, ids, envs, keep_batch = True)
            ));
        for percent in self.threshold_percents:
            self.thresholds[percent] = np.percentile(rcst_errors.cpu(), percent)
            logger.debug(f'Updated threshold({percent:.2f}%): {self.threshold}');
    
    def set_threshold(self, percent):
        self.threshold = self.thresholds[percent];


    def intrusion_detection(self, amps):
        '''
        Rrturns:
        --------
        Is it an intruder.
        '''
        rcst_errors = self.cal_loss(amps, None, None, keep_batch = True);
        return rcst_errors >= self.threshold;

            
