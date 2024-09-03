import torch
from lib import glb_var
import numpy as np
from model.framework.base import Net

device = glb_var.get_value('device');
logger = glb_var.get_value('logger');

class Caution(Net):
    def __init__(self, model_cfg) -> None:
        super().__init__(model_cfg);
        #load and freeze Parameters
        self.pretrained_enc = torch.load(self.pretrained_enc_dir, weights_only=False)['model'];
        for param in self.pretrained_enc.parameters():
            param.requires_grad = False;
        self.pretrained_enc.eval();
        self.is_Intrusion_Detection = True;
        self.threshold = None;

    @torch.no_grad()
    def _cal_R(self, amps):
        #amps:[B, 6000, 3, 56]
        features = self.pretrained_enc.encoder(amps);
        #[B, 256]
        distances = torch.cdist(features, self.pretrained_enc.pcenters);
        min_distances, _ = torch.topk(distances, 2, dim=1, largest=False);
        #rs:[B]
        rs = min_distances[:, 0] / min_distances[:, 1];
        return rs
    
    @torch.no_grad()
    def threshold_update(self, loader):
        rs = torch.zeros(0, device = device);
        intrude_gt = torch.zeros(0, device = device);
        for amps, ids, _ in iter(loader):
            rs = torch.cat((
                rs,
                self._cal_R(amps)
            ));
            intrude_gt = torch.cat((intrude_gt, ids));
        intrude_gt = intrude_gt >= self.known_p_num;
        best_threshold = self.initial_threshold;
        assert 0 <= best_threshold and best_threshold <= 1
        best_accuracy = 0
        for _ in range(self.num_iterations):
            coeff_s = best_threshold - self.threshold_step if best_threshold - self.threshold_step >= 0 else 0;
            coeff_e = best_threshold + self.threshold_step if best_threshold + self.threshold_step <= 1 else 1;
            thresholds = torch.linspace(coeff_s,coeff_e, self.num_thresholds)
            accuracies = [];
            for threshold in thresholds:
                intrude_pred = rs >= threshold;
                accuracy = (intrude_gt == intrude_pred).cpu().float().mean().item();
                accuracies.append(accuracy);
                

            best_idx = np.argmax(accuracies)
            if accuracies[best_idx] > best_accuracy:
                best_accuracy = accuracies[best_idx]
                best_threshold = thresholds[best_idx]
            else:
                break
            logger.info(f'Caution - threshold: {threshold} - acc: {accuracy}');
        self.threshold = best_threshold;

    @torch.no_grad()
    def intrusion_detection(self, amps):
        rs = self._cal_R(amps);
        return rs >= self.threshold;