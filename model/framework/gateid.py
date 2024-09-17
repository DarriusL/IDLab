import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import pairwise_distances
import scipy.stats
import numpy as np
from model.framework.base import Net

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

        # 确保它们在同一设备上
        device = projected_features.device  # 获取 projected_features 的设备

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

        if self.window_size > T:
            raise ValueError(f"Window size {self.window_size} must be less than or equal to T {T}")

        amps_window = amps.unfold(2, self.window_size, self.window_size)

        amps_max = amps_window.max(dim=-1)[0]
        amps_min = amps_window.min(dim=-1)[0]
        amps_mean = amps_window.mean(dim=-1)
        amps_std = amps_window.std(dim=-1)

        amps_window_np = amps_window.cpu().numpy()
        amps_window_np_normalized = (amps_window_np - np.mean(amps_window_np, axis=-1, keepdims=True)) / np.std(
            amps_window_np, axis=-1, keepdims=True)
        amps_skew = torch.tensor(scipy.stats.skew(amps_window_np_normalized, axis=-1), device=amps.device)
        amps_kurtosis = torch.tensor(scipy.stats.kurtosis(amps_window_np_normalized, axis=-1), device=amps.device)

        median_vals = torch.median(amps_window, dim=-1)[0]
        median_deviation_vals = torch.median(torch.abs(amps_window - median_vals.unsqueeze(-1)), dim=-1)[0]

        features = torch.stack([
            amps_max, amps_min, amps_mean,
            amps_std, amps_skew, amps_kurtosis,
            median_deviation_vals
        ], dim=-1)

        return features

# ReliefF算法实现
    def reliefF(self, X, y, num_neighbors=10):
        # 确保 X 是一个 PyTorch 张量
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, device='cuda' if torch.cuda.is_available() else 'cpu')

        batch_size, channels, time_steps, feature_dim, num_features = X.shape

        # 计算每个特征的重要性
        importance_scores = np.zeros((1, num_features))

        for b in range(batch_size):
            for c in range(channels):
                for t in range(time_steps):
                    # 获取当前时间步的所有特征
                    X_np = X[b, c, t].cpu().numpy()  # 这里 X_np 的形状为 (56, 7)

                    # 计算样本之间的距离
                    distances = pairwise_distances(X_np)  # 计算所有样本之间的距离

                    # 对每个样本计算近邻
                    for f in range(feature_dim):
                        sample = X[b, c, t, f].cpu().numpy()  # 将样本移到 CPU 并转换为 NumPy 数组

                        # 获取当前样本的距离
                        sample_distances = distances[f]

                        # 获取最近的同类样本（近邻命中）
                        near_hit_indices = np.argsort(sample_distances)[:num_neighbors]
                        near_hits = X_np[near_hit_indices]

                        # 计算同类样本的差异
                        importance_scores[0, :] -= np.abs(near_hits - sample).sum(axis=0)

                        # 获取最近的异类样本（近邻未命中）
                        near_miss_indices = np.argsort(sample_distances)[num_neighbors:num_neighbors * 2]
                        near_misses = X_np[near_miss_indices]

                        # 计算异类样本的差异
                        importance_scores[0, :] += np.abs(near_misses - sample).sum(axis=0)

        # 归一化重要性分数
        importance_scores /= (num_neighbors * feature_dim * time_steps * channels)

        return torch.from_numpy(importance_scores)  # 输出形状为 (1, 7)

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
        #【B， 3, 6000, 56】
        # 特征融合
        x = x.permute(0, 2, 1, 3);
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
