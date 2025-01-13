import torch
from lib import glb_var, util
from model.framework.base import Net
from model import net_util

class CSIID(Net):
    def __init__(self, model_cfg):
        super().__init__(model_cfg);

        # First convolutional layer
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 30, kernel_size=(100, 3), stride=(1, 3)),
            torch.nn.BatchNorm2d(30),
            torch.nn.ReLU()
        )

        # Second convolutional layer
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(30, 30, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1)),
            torch.nn.BatchNorm2d(30),
            torch.nn.ReLU()
        )

        # Third convolutional layer
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(30, 30, kernel_size=(3, 3), stride=(1, 1), padding=(1, 0)),
            torch.nn.BatchNorm2d(30),
            torch.nn.ReLU()
        )

        # LSTM layer
        self.lstm = torch.nn.LSTM(input_size=480, hidden_size=128, num_layers=5, batch_first=True)

        # Fully connected layer
        self.fc = torch.nn.Linear(128, self.known_p_num)

        # Softmax
        self.softmax = torch.nn.Softmax(dim=1)

    @torch.no_grad()
    def encoder(self, amps):

        x = amps.permute(0, 2, 1, 3)

        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Crop to 203*22*30


        # Reshape for LSTM input (batch_size, seq_len, input_size)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(x.size(0), x.size(1), -1)

        # LSTM layer
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Take the output from the last time step
        return x;

    def p_classifier(self, features):
        x = self.fc(features)

        # Softmax
        x = self.softmax(x)
        return x;

    def p_classify(self, x):
        x = self.encoder(x);
        x = self.p_classifier(x);
        return x
    
    def cal_loss(self, amps, ids, envs):
        id_probs = self.p_classify(amps);
        return torch.nn.CrossEntropyLoss()(id_probs, ids);

class CSIIDAL(CSIID):
    def __init__(self, model_cfg):
        super().__init__(model_cfg);
        self.lambda_ = model_cfg['lambda_'];

        #env layer
        env_cfg = model_cfg['env_classifier'];
        env_cfg['activation_fn'] = net_util.get_activation_fn(env_cfg['activation_fn']);
        env_cfg['dim_in'] = 128;
        env_cfg['dim_out'] = self.known_env_num;
        self.env_layer = util.get_func_rets(net_util.get_mlp_net, env_cfg);

    def cal_loss(self, amps, ids, envs, amps_t = None, ids_t = None, envs_t = None):
        features = self.encoder(amps);
        id_probs = self.p_classifier(features);
        loss_id = torch.nn.CrossEntropyLoss()(id_probs, ids);

        if amps_t is None:
            return loss_id;

        env_probs = self.env_layer(net_util.GradientReversalF.apply(features, self.lambda_));
        loss_env = torch.nn.CrossEntropyLoss()(env_probs, envs);

        loss_t = loss_id + loss_env;

        feature_t = self.encoder(amps_t);
        id_probs_t = self.p_classifier(feature_t);
        id_loss_t = torch.nn.CrossEntropyLoss()(id_probs_t, ids_t);

        return loss_t + id_loss_t;

glb_var.register_model('CSIID', CSIID);
glb_var.register_model('CSIIDAL', CSIIDAL);
