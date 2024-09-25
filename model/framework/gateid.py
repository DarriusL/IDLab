import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import pairwise_distances
import scipy.stats
import numpy as np
from model.framework.base import Net
from lib import glb_var

device = glb_var.get_value('device');

# 特征投影层
class FeatureProjectionLayer(nn.Module):
    def __init__(self):
        super(FeatureProjectionLayer, self).__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=1)  # 1x1卷积Z
        self.gru = nn.GRU(3 * 56 * 7, 56, batch_first=True)  # GRU 输入特征维度为 3 * 56 * 7
        self.fc = nn.Linear(56, 3 * 56 * 7)  # 全连接层，用于将 GRU 输出转换回原始形状

    def forward(self, x):
        # x 的形状为 (batch_size, 3, 150, 56, 7)
        batch_size, channels, height, width, depth = x.size()

        # 通过卷积层
        x_reshaped = x.view(batch_size * height, channels, width, depth)  # 重塑为 (batch_size * 150, 3, 56, 7)
        projected_features = self.conv(x_reshaped)  # 通过卷积层

        # projected_features 的形状为 (2400, 3, 56, 7)
        # 需要将其调整为 (batch_size, height, -1)
        projected_features = projected_features.view(batch_size, height, -1)  # 这里的 -1 会计算为 3 * 56 * 7 = 1176

        # 使用门控线性单元
        output, _ = self.gru(projected_features)  # 直接传递给 GRU

        # 使用全连接层将 GRU 输出转换回原始形状
        output = self.fc(output)  # output 的形状为 (batch_size, height, 3 * 56 * 7)
        output = output.view(batch_size, height, 3, 56, 7)  # 重新调整形状为 (batch_size, 150, 3, 56, 7)

        return output


# 特征融合层
class FeatureFusionLayer(nn.Module):
    def __init__(self, window_size):
        super(FeatureFusionLayer, self).__init__()
        self.window_size = window_size
        self.feature_projection = FeatureProjectionLayer()

    def forward(self, x, y):
        ext_features = self._ext_features(x)

        importance_scores = self.reliefF(ext_features, y)

        projected_features = self.feature_projection(ext_features)

        # 将 importance_scores 移动到同一设备
        importance_scores = importance_scores.to(device)

        # 1. 调整 importance_scores 的形状
        # 需要将其调整为 (1, 1, 1, 7) 以便进行逐元素乘法
        importance_scores = importance_scores.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # 变为 (1, 1, 1, 7)

        # 2. 使用 importance_scores 对 projected_features 进行加权
        weighted_features = projected_features * importance_scores  # 逐元素乘法

        # 3. 对最后一个维度进行求和以获得最终输出
        weighted_sum = torch.sum(weighted_features, dim=-1)  # 在最后一个维度上求和
        return weighted_sum

    def _ext_features(self, amps):
        B, R, T, F = amps.shape

        # if self.window_size > T:
        #     raise ValueError(f"Window size {self.window_size} must be less than or equal to T {T}")

        amps_window = amps.unfold(2, self.window_size, self.window_size)

        amps_max = amps_window.max(dim=-1)[0]
        amps_min = amps_window.min(dim=-1)[0]
        amps_mean = amps_window.mean(dim=-1)
        amps_std = amps_window.std(dim=-1)

        amps_window_np = amps_window.cpu().numpy()
        amps_window_np_normalized = (amps_window_np - np.mean(amps_window_np, axis=-1, keepdims=True)) / np.std(
            amps_window_np, axis=-1, keepdims=True)
        amps_skew = torch.tensor(scipy.stats.skew(amps_window_np_normalized, axis=-1), device=device)
        amps_kurtosis = torch.tensor(scipy.stats.kurtosis(amps_window_np_normalized, axis=-1), device=device)

        median_vals = torch.median(amps_window, dim=-1)[0]
        median_deviation_vals = torch.median(torch.abs(amps_window - median_vals.unsqueeze(-1)), dim=-1)[0]

        features = torch.stack([
            amps_max, amps_min, amps_mean,
            amps_std, amps_skew, amps_kurtosis,
            median_deviation_vals
        ], dim=-1)

        return features

    def reliefF(self, X, y, num_neighbors=10):
        """
        ReliefF 算法，用于计算特征的重要性分数。
        
        参数:
            X: 输入数据，形状为 (batch_size, channels, time_steps, feature_dim, num_features) 的张量。
            y: 标签（未在此函数中使用，但可能需要在其他版本中考虑）。
            num_neighbors: 每个样本的近邻数量。
        
        返回:
            重要性分数，形状为 (1, num_features) 的张量。
        """
        batch_size, channels, time_steps, feature_dim, num_features = X.shape

        # 初始化重要性分数
        importance_scores = torch.zeros((1, num_features), device=X.device)

        # 将 (batch_size, channels, time_steps, feature_dim, num_features) 展平为 (total_samples, feature_dim, num_features)
        total_samples = batch_size * channels * time_steps
        X_flat = X.view(total_samples, feature_dim, num_features)  # 形状为 (total_samples, feature_dim, num_features)

        # 对每个时间步的样本计算距离矩阵，形状为 (total_samples, feature_dim, feature_dim)
        distances = torch.zeros((total_samples, feature_dim, feature_dim), device=X.device)
        for i in range(total_samples):
            distances[i] = self.compute_pairwise_distances(X_flat[i])

        # 获取每个样本的最近的同类样本（近邻命中），形状为 (total_samples, feature_dim, num_neighbors)
        near_hit_indices = torch.argsort(distances, dim=2)[:, :, :num_neighbors]
        near_hits = torch.gather(X_flat.unsqueeze(2).expand(-1, -1, num_neighbors, -1), 1, near_hit_indices.unsqueeze(-1).expand(-1, -1, -1, num_features))

        # 计算同类样本的差异，形状为 (total_samples, feature_dim, num_features)
        hit_differences = torch.abs(near_hits - X_flat.unsqueeze(2)).sum(dim=2)

        # 获取每个样本的最近的异类样本（近邻未命中），形状为 (total_samples, feature_dim, num_neighbors)
        near_miss_indices = torch.argsort(distances, dim=2)[:, :, num_neighbors:num_neighbors * 2]
        near_misses = torch.gather(X_flat.unsqueeze(2).expand(-1, -1, num_neighbors, -1), 1, near_miss_indices.unsqueeze(-1).expand(-1, -1, -1, num_features))

        # 计算异类样本的差异，形状为 (total_samples, feature_dim, num_features)
        miss_differences = torch.abs(near_misses - X_flat.unsqueeze(2)).sum(dim=2)

        # 更新重要性分数
        importance_scores[0] -= hit_differences.sum(dim=1).sum(dim=0)
        importance_scores[0] += miss_differences.sum(dim=1).sum(dim=0)

        # 归一化重要性分数
        importance_scores /= (num_neighbors * feature_dim * time_steps * channels)

        return importance_scores  # 输出形状为 (1, num_features)


    def compute_pairwise_distances(self, X_tensor):
        # 计算每一对样本之间的欧几里得距离
        X_norm = (X_tensor ** 2).sum(dim=1, keepdim=True)  # 每个样本的平方和
        distances = X_norm + X_norm.T - 2 * torch.mm(X_tensor, X_tensor.T)  # 计算距离
        return distances

# ResNet块
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        if x.dtype != self.conv1.weight.dtype:
            x = x.to(self.conv1.weight.dtype)
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# 身份识别模型
class GateID(Net):
    def __init__(self, model_cfg):
        super().__init__(model_cfg);
        self.feature_fusion = FeatureFusionLayer(window_size=self.window_size)

        # ResNet部分
        self.resnet_layer1 = ResNetBlock(3, 64)  # 输入通道为3
        self.resnet_layer2 = ResNetBlock(64, 128)

        self.conv_layer = nn.Conv2d(128, 128, kernel_size=3, padding=1)  # 3x3卷积层
        self.dropout = nn.Dropout(0.5)  # 丢弃层

        # Bi-LSTM部分
        self.lstm = nn.LSTM(128, 64, bidirectional=True, batch_first=True)

        # 线性层
        self.fc = nn.Linear(64 * 2, self.known_p_num)  # 双向LSTM输出维度为64*2

    def p_classify(self, x, y):
        x = x.permute(0, 2, 1, 3);
        # 特征融合
        x = self.feature_fusion(x, y)

        # 调整形状以适应 ResNet
        x = x.permute(0, 2, 1, 3)  # 从 [16, 150, 3, 56] 变为 [16, 3, 150, 56]

        # ResNet特征提取
        x = self.resnet_layer1(x)
        x = self.resnet_layer2(x)

        # 特征压缩
        x = self.conv_layer(x)
        x = F.relu(x)
        x = self.dropout(x)

        # LSTM处理
        x = x.view(x.size(0), -1, 128)  # 假设每个时间步有128个特征
        x, _ = self.lstm(x)

        # 线性层预测
        x = self.fc(x[:, -1, :])  # 取最后一个时间步的输出
        return x
    
    def cal_loss(self, amps, ids, envs, is_target_data = False):
        id_probs = self.p_classify(amps, ids);
        return torch.nn.CrossEntropyLoss()(id_probs, ids);

    @torch.no_grad()
    def cal_accuracy(self, amps, ids, envs):
        #ids:[N]
        #envs:[N]

            #p
        id_pred = self.p_classify(amps, ids).argmax(dim = -1);
        acc = (id_pred == ids).cpu().float().mean().item();
        return acc;
