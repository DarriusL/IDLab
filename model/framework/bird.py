#Bi-Stage Identity Recognition and Intrusion Detection
import torch, sklearn.svm, sklearn.preprocessing,joblib
import numpy as np
from lib import util, glb_var
from model import net_util
from model.framework.base import Net
from model import attnet

logger = glb_var.get_value('logger');
device = glb_var.get_value('device');

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
        #input:[N, 1, t, d]
        return input + self.net(input);

class CSINet(torch.nn.Module):
    def __init__(self, refinenet_cfg) -> None:
        super().__init__();
        layers = [
            RefineNet(refinenet_cfg) for _ in range(refinenet_cfg['n_refine'])
        ]
        self.net = torch.nn.Sequential(*layers);

    def forward(self, input):
        #input:[N, t, 1, d] -> [N, 1, t, d]
        #output:[N, 1, t, d] -> [N, t, d]
        return self.net(input.permute(0, 2, 1, 3)).squeeze(1);

class BIRDEncoder(Net):
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
        csinet_cfg = model_cfg['csinet'];
        csinet_cfg['in_channels'] = 1;
        csinet_cfg['shape'] = self.t;
        self.csinets = torch.nn.ModuleList([
            CSINet(csinet_cfg) for _ in range(self.dim_in[2])
        ])

        #trans
        trans2_cfg = model_cfg['trans2'];
        trans2_cfg['d'] = self.d;
        self.trans2 = Trans(trans2_cfg);
        
        #tail_net
        tail_net_cfg = model_cfg['tail_net'];
        tail_net_cfg['dim_in'] = self.dim_in[2] * self.t * self.d;
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
        #([N, t, d], [N, t, d], [N, t, d])
        aa1 = tuple([self.csinets[i](amps2[:, :, [i], :]) for i in range(self.dim_in[2])]);
        #[N, t*3, d]
        te1 = torch.cat(aa1, dim = 1);
        #[N, t*3, d]
        te2 = self.trans2(te1);
        #[N, 256]
        feature = self.tail_net(te2.reshape(amps.shape[0], -1));
        return feature;

    def p_classify(self, amps):
        return self.p_classifier(self.encoder(amps));

    def cal_loss(self, amps, ids, envs, amps_t = None, ids_t = None, envs_t = None):
        #amps:[N, T, R, F]
        #ids:[N]
        #envs:[N]
        feature = self.encoder(amps);
        id_probs = self.p_classifier(feature);
        id_loss = torch.nn.CrossEntropyLoss()(id_probs, ids);
        env_probs = self.env_classifier(net_util.GradientReversalF.apply(feature, self.lambda_));
        env_loss = torch.nn.CrossEntropyLoss()(env_probs, envs);
        src_loss = id_loss + env_loss;
        if amps_t is None:
            return src_loss;
        feature_t = self.encoder(amps_t);
        id_probs_t = self.p_classifier(feature_t);
        id_loss_t = torch.nn.CrossEntropyLoss()(id_probs_t, ids_t);

        return src_loss + id_loss_t;

class BIRD(Net):
    def __init__(self, model_cfg) -> None:
        super().__init__(model_cfg);
        #load and freeze Parameters
        from model import load_model
        self.pretrained_enc = load_model(self.pretrained_enc_dir);
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
    
    @torch.no_grad()
    def encoder(self, amps):
        return self.pretrained_enc.encoder(amps)
    
    def reconstruct(self, amps):
        with torch.no_grad():
            #[N, T, R, F] -> [N, do] -> [N, 1, d, d]
            feature1 = self.encoder(amps).reshape(-1, 1, self.d, self.d);
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

class BIRDEncoderSVM(Net):
    def __init__(self, model_cfg) -> None:
        super().__init__(model_cfg);
        from model import load_model
        self.pretrained_enc = load_model(self.pretrained_enc_dir);
        assert self.do == self.pretrained_enc.do;

        self.detector = sklearn.svm.OneClassSVM(kernel = 'rbf', gamma = 'auto', cache_size = 1024, nu = self.abnormality_rate);

        self.is_Intrusion_Detection = True;

    @torch.no_grad()
    def encoder(self, amps):
        return self.pretrained_enc.encoder(amps);

    def conventional_train(self, X, Y):
        X = sklearn.preprocessing.StandardScaler().fit_transform(X);
        self.detector.fit(X);

    def intrusion_detection(self, amps):
        return self.detector.predict(self.encoder(amps).detach().cpu().numpy()) == -1;

    def save(self, save_dir):
        super().save(save_dir);
        #save sklearn model
        joblib.dump(self.detector, save_dir + 'model.pkl');
        logger.info(f'Sklearn model saved to {save_dir}');

    def load(self, load_dir):
        super().load(load_dir)
        self.detector = joblib.load(load_dir + 'model.pkl');
        logger.info(f'Sklearn model saved to {load_dir}');

glb_var.register_model('BIRDEncoder', BIRDEncoder);
glb_var.register_model('BIRD', BIRD);
glb_var.register_model('BIRDEncoderSVM', BIRDEncoderSVM);
