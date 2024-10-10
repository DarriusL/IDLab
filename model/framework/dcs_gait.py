import torch
from model.framework.base import Net
from model import net_util, attnet
from lib import util, glb_var

logger = glb_var.get_value('logger')

@torch.no_grad()
def calculate_distance_matrix(source_features, target_features):
    source_features = source_features.flatten(start_dim=1)
    target_features = target_features.flatten(start_dim=1)
    
    # (source_features.size(0), target_features.size(0))
    cosine_sim = torch.mm(source_features, target_features.T) / (
        torch.norm(source_features, dim=1).unsqueeze(1) * torch.norm(target_features, dim=1).unsqueeze(0)
    )
    
    return 1 - cosine_sim
    
class DCSMultiHeadAttention(Net):
    def __init__(self, model_cfg) -> None:
        super().__init__(model_cfg)
        self.net = attnet.MultiHeadAttention(self.d, self.d_q, self.d_k, self.d_v, self.n_heads);

    def forward(self, src = None, tgt = None):
        src = src if src is not None else tgt
        tgt = tgt if tgt is not None else src
        return self.net(src, tgt, tgt).squeeze(1);

class DCSGaitEncoder(Net):
    def __init__(self, model_cfg) -> None:
        super().__init__(model_cfg);
        hid_layer = [self.dim_in[2]] +  model_cfg['cnn_cfg']['hid_layers'] + [self.dim_in[2]];
        pool = torch.nn.MaxPool2d(2, 2);
        f_ext_layers = [
            torch.nn.Sequential(
                torch.nn.Conv2d(hid_layer[i], hid_layer[i + 1], kernel_size = 3, stride = 1, padding = 1),
                pool
            )
            for i in range(len(hid_layer) - 1)
        ] + [torch.nn.Flatten()];
        
        #tail_net
        tail_net_cfg = model_cfg['tail_net_cfg'];
        tail_net_cfg['dim_in'] = ( self.dim_in[1] // 2 ** (len(hid_layer) - 1) ) * ( self.dim_in[-1] // 2 ** (len(hid_layer) - 1) ) * self.dim_in[2];
        tail_net_cfg['dim_out'] = self.do;
        tail_net_cfg['activation_fn'] = net_util.get_activation_fn(tail_net_cfg['activation_fn']);
        f_ext_layers += [util.get_func_rets(net_util.get_mlp_net, tail_net_cfg)];

        self.feature_net = torch.nn.Sequential(*f_ext_layers);

        #p classifier
        p_c_cfg = model_cfg['p_classifier'];
        p_c_cfg['activation_fn'] = net_util.get_activation_fn(p_c_cfg['activation_fn']);
        p_c_cfg['dim_in'] = self.do;
        p_c_cfg['dim_out'] = self.known_p_num;
        self.p_classifier = util.get_func_rets(net_util.get_mlp_net, p_c_cfg);

        self.is_Intrusion_Detection = False;

    def encoder(self, amps):
        #amps:[B, T, R, F]
        return self.feature_net(amps.permute(0, 2, 1, 3));

    def p_classify(self, amps):
        return self.p_classifier(self.encoder(amps));

    def cal_loss(self, amps, ids, envs, amps_t=None, ids_t=None, envs_t=None):
        id_probs = self.p_classify(amps);
        return torch.nn.CrossEntropyLoss()(id_probs, ids);

class DCSGait(Net):
    def __init__(self, model_cfg) -> None:
        super().__init__(model_cfg);

        #load and freeze Parameters
        from model import load_model
        self.pretrained_enc = load_model(self.pretrained_enc_dir);
        for param in self.pretrained_enc.parameters():
            param.requires_grad = False;
        #MultiHeadAttention
        MHA_cfg = model_cfg['MHA_cfg'];
        MHA_cfg['d'] = self.pretrained_enc.do;
        self.MHA = DCSMultiHeadAttention(MHA_cfg)
        
        #p classifier
        p_c_cfg = model_cfg['p_classifier'];
        p_c_cfg['activation_fn'] = net_util.get_activation_fn(p_c_cfg['activation_fn']);
        p_c_cfg['dim_in'] = self.pretrained_enc.do;
        p_c_cfg['dim_out'] = self.known_p_num;
        self.p_classifier = util.get_func_rets(net_util.get_mlp_net, p_c_cfg);
    
    @torch.no_grad()
    def generate_pseudo_labels(self, target_data):
        features = self.encoder(target_data)  # (N, k)

        centers = self.compute_initial_centers(features)  # (k, k)

        for _ in range(self.num_iterations):

            pseudo_labels = self.assign_pseudo_labels(features, centers)

            centers = self.update_centers(features, pseudo_labels, centers)
        
        logger.info('Generating pseudo labels complete.');

        return pseudo_labels

    @torch.no_grad()
    def match_filter(self, source_data, target_data):
        features_t = self.encoder(target_data);
        del target_data

        features_s = [];
        while True:
            features_s.append(self.encoder(source_data[[0], :]));
            if source_data.shape[0] == 1: del source_data; break;
            source_data = source_data[1:, :];
        features_s = torch.stack(features_s);

        distance_matrix_s_to_t = calculate_distance_matrix(features_s, features_t)
        distance_matrix_t_to_s = calculate_distance_matrix(features_t, features_s)

        min_indices_s_to_t = torch.argmin(distance_matrix_s_to_t, dim=1)  # source -> target
        min_indices_t_to_s = torch.argmin(distance_matrix_t_to_s, dim=1)  # target -> source

        logger.info('data matching complete.')
        return min_indices_s_to_t, min_indices_t_to_s

    @torch.no_grad()
    def generate_src_and_tgt(self, amps_s, ids_s, amps_t):
        ids_t = self.generate_pseudo_labels(amps_t);
        idxs_s2t, idxs_t2s = self.match_filter(amps_s, amps_t);
        logger.debug(f'idxs_s2t.shape:{idxs_s2t.shape}, idxs_t2s.shape:{idxs_t2s.shape}');
        amps_s = torch.cat((amps_s, amps_s[idxs_t2s]), dim = 0);
        logger.debug('amps_s.shape:', amps_s.shape);
        ids_so = torch.cat((ids_s, ids_t), dim = 0);
        logger.debug('ids_so.shape: ', ids_so.shape);
        amps_t = torch.cat((amps_t[idxs_s2t], amps_t), dim = 0);
        logger.debug('amps_t.shape:', amps_t.shape)
        ids_to = torch.cat((ids_s, ids_t), dim = 0);
        logger.debug('ids_to.shape:', ids_to.shape);
        return amps_s, ids_so, amps_t, ids_to

    @torch.no_grad()
    def compute_initial_centers(self, features):
        probs = torch.nn.functional.softmax(features, dim=1)  # shape: [N, K]
        centers = torch.sum(features.unsqueeze(1) * probs.unsqueeze(2), dim=0) / torch.sum(probs, dim=0).unsqueeze(1)  # (k,k)
        return centers

    @torch.no_grad()
    def assign_pseudo_labels(self, features, centers):
        distances = torch.cdist(features, centers)
        pseudo_labels = torch.argmin(distances, dim=1)
        return pseudo_labels

    @torch.no_grad()
    def update_centers(self, features, pseudo_labels, centers):
        new_centers = []
        for k in range(self.known_p_num):
            mask = (pseudo_labels == k)
            if mask.sum() > 0:
                new_center = features[mask].mean(dim=0)
            else:
                new_center = centers[k]
            new_centers.append(new_center)
        return torch.stack(new_centers)

    @torch.no_grad()
    def encoder(self, amps):
        return self.pretrained_enc.encoder(amps);

    def forward(self, f_s = None, f_t = None):
        return self.p_classifier(self.MHA(src = f_s, tgt = f_t));

    def p_classify(self, amps_s):
        feature_s = self.encoder(amps_s)
        return self.forward(feature_s);

    def cal_loss(self, amps, ids, envs, amps_t = None, ids_t = None, envs_t = None):
        #src
        feature_s = self.encoder(amps);
        ids_s_probs = self.forward(feature_s);
        loss_id_s = torch.nn.CrossEntropyLoss()(ids_s_probs, ids);

        if amps_t is None:
            return loss_id_s
        
        #tgt
        feature_t = self.encoder(amps_t);
        ids_t_probs = self.forward(f_t = feature_t);
        loss_id_t = torch.nn.CrossEntropyLoss()(ids_t_probs, ids_t);

        #cross
        ids_probs = self.forward(f_s = feature_s, f_t = feature_t);
        loss_cross = torch.nn.CrossEntropyLoss()(ids_probs, ids)
        return loss_id_s + self.alpha * loss_id_t + loss_cross;

glb_var.register_model('DCSGaitEncoder', DCSGaitEncoder);
glb_var.register_model('DCSGait', DCSGait);