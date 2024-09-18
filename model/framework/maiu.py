import torch
from model.framework.base import Net


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

class ResNetUnit(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetUnit, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        self.conv3 = torch.nn.Conv2d(out_channels, out_channels * 4, kernel_size=1)
        self.bn3 = torch.nn.BatchNorm2d(out_channels * 4)

        self.relu = torch.nn.ReLU(inplace=True)
        self.downsample = torch.nn.Sequential()
        if stride != 1 or in_channels != out_channels * 4:
            self.downsample = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=stride),
                torch.nn.BatchNorm2d(out_channels * 4)
            )

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += residual
        out = self.relu(out)
        return out


class MAIU(Net):
    def __init__(self, model_cfg):
        super().__init__(model_cfg);
        self.cnn_unit = ConvBlock(3, 64, kernel_size=7, stride=2, padding=3)

        self.resnet_layers = torch.nn.Sequential(
            ResNetUnit(64, 32, stride=1),
            ResNetUnit(128, 32, stride=1),
            ResNetUnit(128, 64, stride=2),
            ResNetUnit(256, 128, stride=2),
            ResNetUnit(512, 256, stride=2)
        )

        self.global_avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc_identity = torch.nn.Linear(1024, self.known_p_num);

    def p_classify(self, x):
        x = x.permute(0, 2, 1, 3)
        x = self.cnn_unit(x)
        x = self.resnet_layers(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        identity_out = self.fc_identity(x)
        return identity_out;

    def cal_loss(self, amps, ids, envs, is_target_data = False):
        id_probs = self.p_classify(amps);
        return torch.nn.CrossEntropyLoss()(id_probs, ids);


class MAIUId(Net):
    def __init__(self, model_cfg):
        super().__init__(model_cfg);
        self.cnn_unit = ConvBlock(3, 64, kernel_size=7, stride=2, padding=3)

        self.resnet_layers = torch.nn.Sequential(
            ResNetUnit(64, 32, stride=1),
            ResNetUnit(128, 32, stride=1),
            ResNetUnit(128, 64, stride=2),
            ResNetUnit(256, 128, stride=2),
            ResNetUnit(512, 256, stride=2)
        )

        self.global_avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc_identity = torch.nn.Linear(1024, self.known_p_num)
        self.fc_illegal = torch.nn.Linear(1024, 2)

        self.is_Intrusion_Detection = True;

    def intrusion_detection(self, amps):
        '''
        Rrturns:
        --------
        Is it an intruder.
        '''
        x = amps.permute(0, 2, 1, 3)
        x = self.cnn_unit(x)
        x = self.resnet_layers(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc_illegal(x).argmax(dim = -1) == 1;

    def cal_loss(self, amps, ids, envs, is_target_data = False):
        x = amps.permute(0, 2, 1, 3)
        x = self.cnn_unit(x)
        x = self.resnet_layers(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        identity_out = self.fc_identity(x)
        illegal_out = self.fc_illegal(x)
        #loss1
        loss1 = self.cross_entropy(identity_out, ids)
        l2_reg = sum(p.pow(2.0).sum() for p in self.parameters())
        loss1 += self.lambda_reg * l2_reg

        #loss2
        illegal_target = (ids > self.illegal_target).int();
        illegal_probs = torch.nn.functional.softmax(illegal_out, dim=1).argmax(dim=1)  # 预测标签
        indicator = ((illegal_probs == 0) & (illegal_target == 0)).float()  # 0表示合法用户

        max_term = torch.clamp(indicator - illegal_probs + illegal_probs / (self.p * self.d), 0)
        loss2 = torch.mean(max_term)
        return loss1 + loss2

