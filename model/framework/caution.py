import torch, tqdm, sys
from lib import glb_var, util, callback
import numpy as np
from model.framework.base import Net
from model import net_util

device = glb_var.get_value('device');
logger = glb_var.get_value('logger');

class CautionEncoder(Net):
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
        self.remind = False;
        self.is_Intrusion_Detection = False
    
    def encoder(self, amps):
        #amps:[B, 6000, 3, 56] -> [B, 3, 6000, 56]
        feature1 = self.cnn(amps.permute(0, 2, 1, 3));
        feature2 = self.FeatureLayer(feature1.reshape(amps.shape[0], -1));
        #[B, do]
        return feature2;

    @torch.no_grad()
    def update_center(self, support_loader):
        self.eval();
        self.pcenters = torch.zeros((self.known_p_num, self.do), device = device);
        p_id = 0;
        for amps, ids, _ in tqdm.tqdm(support_loader, desc=f"Updateing center of each class", unit="batch", leave=False, file=sys.stdout):
            with callback.no_stdout():
            #amps:[1, B, 6000, 3, 56]
                assert torch.all(ids == p_id);
                #[B, 6000, 3, 56] -> [B, do] -> [do]
                self.pcenters[p_id, :] = self.encoder(amps.squeeze(0)).mean(dim = 0); 
                p_id += 1;
    
    def p_classify(self, amps):
        #[B, do]
        features = self.encoder(amps);
        #[B, N]
        distences = torch.exp( - torch.cdist(features, self.pcenters));
        #[B]
        probs = distences / distences.sum(dim = 1, keepdim = True);
        return probs;

    def cal_loss(self, amps, ids, envs):
        probs = self.p_classify(amps);
        return torch.nn.CrossEntropyLoss()(probs, ids);

    def train_epoch_hook(self, trainer, epoch):
        self.eval()
        if ((epoch + 1) >= self.update_start_epoch and (epoch + 1) % self.update_step == 0) or epoch == 0:
            self.update_center(trainer.support_loader);
            logger.info('update each class center completed.');
        self.train();

    def pre_test_hook(self, tester):
        self.update_center(tester.support_loader);


class Caution(Net):
    def __init__(self, model_cfg) -> None:
        super().__init__(model_cfg);
        from model import load_model
        #load and freeze Parameters
        self.pretrained_enc = load_model(self.pretrained_enc_dir);
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
    
    def pre_test_hook(self, tester):
        self.pretrained_enc.update_center(tester.support_loader);
        self.threshold_update(tester.valid_loader);

    @torch.no_grad()
    def intrusion_detection(self, amps):
        rs = self._cal_R(amps);
        return rs >= self.threshold;

glb_var.register_model('CautionEncoder', CautionEncoder);
glb_var.register_model('Caution', Caution)
