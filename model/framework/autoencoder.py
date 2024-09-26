import torch
from lib import util, glb_var
import numpy as np
from model import net_util
from model.framework.base import Net

device = glb_var.get_value('device');
logger = glb_var.get_value('logger');

class AE(Net):
    def __init__(self, model_cfg) -> None:
        super().__init__(model_cfg);
        
        #encoder
        encoder_cfg = model_cfg['encoder_cfg'];
        encoder_cfg['activation_fn'] = net_util.get_activation_fn(encoder_cfg['activation_fn']);
        encoder_cfg['dim_in'] = np.prod(self.dim_in[1:]);
        encoder_cfg['dim_out'] = self.do;
        self.encoder = util.get_func_rets(net_util.get_mlp_net, encoder_cfg);

        #decoder
        decoder_cfg = model_cfg['decoder_cfg'];
        decoder_cfg['activation_fn'] = net_util.get_activation_fn(decoder_cfg['activation_fn']);
        decoder_cfg['dim_in'] = self.do;
        decoder_cfg['dim_out'] = np.prod(self.dim_in[1:]);
        self.decoder = util.get_func_rets(net_util.get_mlp_net, decoder_cfg);

        self.loss_func = net_util.get_loss_func(model_cfg['loss']);
        self.threshold = .0;
        self.thresholds = dict();

        self.is_Intrusion_Detection = True;

    def reconstruct(self, amps):
        #amps[B, T, R, F]
        feature = self.encoder(amps.flatten(1, -1));
        csi_rcst = self.decoder(feature);
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
