import torch
from lib import glb_var, callback

logger = glb_var.get_value('logger');

def get_mlp_net(hid_layers, activation_fn, drop_out, dim_in, dim_out, end_with_softmax = False):
    ''''''
    if len(hid_layers) > 1:
        layers = [
            torch.nn.Linear(dim_in, hid_layers[0]),
            torch.nn.Dropout(drop_out),
            activation_fn] + [
            torch.nn.Linear(hid_layers[i], hid_layers[i+1]) for i in range(len(hid_layers) - 1)] + [
            activation_fn,
            torch.nn.Linear(hid_layers[-1], dim_out)    
        ];
    elif len(hid_layers) == 1:
        layers = [
            torch.nn.Linear(dim_in, hid_layers[0]),
            torch.nn.Dropout(drop_out),
            activation_fn,
            torch.nn.Linear(hid_layers[0], dim_out)
        ]
    else:
        #len()==0
        layers = [
            torch.nn.Linear(dim_in, dim_out)
        ]
    
    if end_with_softmax:
        layers = layers + [torch.nn.Softmax(dim = -1)];
    return torch.nn.Sequential(*layers);

def get_activation_fn(name = 'selu'):
    '''
    Get the activation function

    Parameters:
    ----------
    name: str
        the name of the activation function
        default: 'selu'
    '''
    activations = ['sigmoid', 'tanh', 'relu', 'leaky_relu', 'elu', 'softmax', 'selu'];
    if name.lower() == 'sigmoid':
        return torch.nn.Sigmoid();
    elif name.lower() == 'tanh':
        return torch.nn.Tanh();
    elif name.lower() == 'relu':
        return torch.nn.ReLU();
    elif name.lower() == 'leaky_relu':
        return torch.nn.LeakyReLU();
    elif name.lower() == 'elu':
        return torch.nn.ELU();
    elif name.lower() == 'softmax':
        return torch.nn.Softmax();
    elif name.lower() == 'selu':
        return torch.nn.SELU();
    elif name.lower() == 'mish':
        return torch.nn.Mish();
    else:
        logger.error(f'Activation function [{name.lower()}] does not support automatic acquisition at the moment,'
                                        f'please replace or add the code yourself.\nSupport list:{activations}');
        raise callback.CustomException('ActivationCfgNameError');


class GradualWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, multiplier, warm_epoch, after_scheduler=None):
        self.multiplier = multiplier;
        self.total_epoch = warm_epoch;
        self.after_scheduler = after_scheduler;
        self.finished = False;
        self.last_epoch = None;
        self.base_lrs = None;
        super().__init__(optimizer);

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs];
                    self.finished = True;
                return self.after_scheduler.get_last_lr();
            return [base_lr * self.multiplier for base_lr in self.base_lrs];
        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs];


    def step(self, epoch=None, metrics=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None);
            else:
                self.after_scheduler.step(epoch - self.total_epoch);
        else:
            return super(GradualWarmupScheduler, self).step(epoch);

class AugmentedConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dk, dv, Nh, shape=0, relative=False, stride=1):
        super(AugmentedConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dk = dk
        self.dv = dv
        self.Nh = Nh
        self.shape = shape
        self.relative = relative
        self.stride = stride
        self.padding = (self.kernel_size - 1) // 2

        assert self.Nh != 0, "integer division or modulo by zero, Nh >= 1"
        assert self.dk % self.Nh == 0, "dk should be divided by Nh. (example: out_channels: 20, dk: 40, Nh: 4)"
        assert self.dv % self.Nh == 0, "dv should be divided by Nh. (example: out_channels: 20, dv: 4, Nh: 4)"
        assert stride in [1, 2], str(stride) + " Up to 2 strides are allowed."

        self.conv_out = torch.nn.Conv2d(self.in_channels, self.out_channels - self.dv, self.kernel_size, stride=stride, padding=self.padding)

        self.qkv_conv = torch.nn.Conv2d(self.in_channels, 2 * self.dk + self.dv, kernel_size=self.kernel_size, stride=stride, padding=self.padding)

        self.attn_out = torch.nn.Conv2d(self.dv, self.dv, kernel_size=1, stride=1)

        if self.relative:
            self.key_rel_w = torch.nn.Parameter(torch.randn((2 * self.shape - 1, dk // Nh), requires_grad=True))
            self.key_rel_h = torch.nn.Parameter(torch.randn((2 * self.shape - 1, dk // Nh), requires_grad=True))

    def forward(self, x):
        # Input x
        # (batch_size, channels, height, width)
        # batch, _, height, width = x.size()

        # conv_out
        # (batch_size, out_channels, height, width)
        conv_out = self.conv_out(x)
        batch, _, height, width = conv_out.size()

        # flat_q, flat_k, flat_v
        # (batch_size, Nh, height * width, dvh or dkh)
        # dvh = dv / Nh, dkh = dk / Nh
        # q, k, v
        # (batch_size, Nh, height, width, dv or dk)
        flat_q, flat_k, flat_v, q, k, v = self.compute_flat_qkv(x, self.dk, self.dv, self.Nh)
        logits = torch.matmul(flat_q.transpose(2, 3), flat_k)
        if self.relative:
            h_rel_logits, w_rel_logits = self.relative_logits(q)
            logits += h_rel_logits
            logits += w_rel_logits
        weights = torch.nn.functional.softmax(logits, dim=-1)

        # attn_out
        # (batch, Nh, height * width, dvh)
        attn_out = torch.matmul(weights, flat_v.transpose(2, 3))
        attn_out = torch.reshape(attn_out, (batch, self.Nh, self.dv // self.Nh, height, width))
        # combine_heads_2d
        # (batch, out_channels, height, width)
        attn_out = self.combine_heads_2d(attn_out)
        attn_out = self.attn_out(attn_out)
        return torch.cat((conv_out, attn_out), dim=1)

    def compute_flat_qkv(self, x, dk, dv, Nh):
        qkv = self.qkv_conv(x)
        N, _, H, W = qkv.size()
        q, k, v = torch.split(qkv, [dk, dk, dv], dim=1)
        q = self.split_heads_2d(q, Nh)
        k = self.split_heads_2d(k, Nh)
        v = self.split_heads_2d(v, Nh)

        dkh = dk // Nh
        q = q * dkh ** -0.5
        flat_q = torch.reshape(q, (N, Nh, dk // Nh, H * W))
        flat_k = torch.reshape(k, (N, Nh, dk // Nh, H * W))
        flat_v = torch.reshape(v, (N, Nh, dv // Nh, H * W))
        return flat_q, flat_k, flat_v, q, k, v

    def split_heads_2d(self, x, Nh):
        batch, channels, height, width = x.size()
        ret_shape = (batch, Nh, channels // Nh, height, width)
        split = torch.reshape(x, ret_shape)
        return split

    def combine_heads_2d(self, x):
        batch, Nh, dv, H, W = x.size()
        ret_shape = (batch, Nh * dv, H, W)
        return torch.reshape(x, ret_shape)

    def relative_logits(self, q):
        B, Nh, dk, H, W = q.size()
        q = torch.transpose(q, 2, 4).transpose(2, 3)

        rel_logits_w = self.relative_logits_1d(q, self.key_rel_w, H, W, Nh, "w")
        rel_logits_h = self.relative_logits_1d(torch.transpose(q, 2, 3), self.key_rel_h, W, H, Nh, "h")

        return rel_logits_h, rel_logits_w

    def relative_logits_1d(self, q, rel_k, H, W, Nh, case):
        rel_logits = torch.einsum('bhxyd,md->bhxym', q, rel_k)
        rel_logits = torch.reshape(rel_logits, (-1, Nh * H, W, 2 * W - 1))
        rel_logits = self.rel_to_abs(rel_logits)

        rel_logits = torch.reshape(rel_logits, (-1, Nh, H, W, W))
        rel_logits = torch.unsqueeze(rel_logits, dim=3)
        rel_logits = rel_logits.repeat((1, 1, 1, H, 1, 1))

        if case == "w":
            rel_logits = torch.transpose(rel_logits, 3, 4)
        elif case == "h":
            rel_logits = torch.transpose(rel_logits, 2, 4).transpose(4, 5).transpose(3, 5)
        rel_logits = torch.reshape(rel_logits, (-1, Nh, H * W, H * W))
        return rel_logits

    def rel_to_abs(self, x):
        B, Nh, L, _ = x.size()

        col_pad = torch.zeros((B, Nh, L, 1)).to(x)
        x = torch.cat((x, col_pad), dim=3)

        flat_x = torch.reshape(x, (B, Nh, L * 2 * L))
        flat_pad = torch.zeros((B, Nh, L - 1)).to(x)
        flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)

        final_x = torch.reshape(flat_x_padded, (B, Nh, L + 1, 2 * L - 1))
        final_x = final_x[:, :, :L, L - 1:]
        return final_x
    
class GradientReversalF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_;

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambda_

        return output, None
    
def get_free_gpu_memory(device):
    '''
    Retrieve the remaining GPU memory of the device(GB)
    '''
    if not torch.cuda.is_available():
        return .0;
    total_memory = torch.cuda.get_device_properties(device).total_memory
    reserved_memory = torch.cuda.memory_reserved(device)
    free_memory = (total_memory - reserved_memory) / 1024**3;
    return free_memory;

class MSELoss(torch.nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.mse = torch.nn.MSELoss()
        self.mse_none = torch.nn.MSELoss(reduction = 'none');
    
    def forward(self, input, target, keep_batch = False):
        if not keep_batch:
            return self.mse(input, target);
        else:
            return self.mse_none(input, target).mean(dim = [1, 2, 3]);

class NMSELoss(torch.nn.Module):
    def __init__(self):
        super(NMSELoss, self).__init__()
        self.mse = torch.nn.MSELoss()
        self.mse_none = torch.nn.MSELoss(reduction = 'none');
    
    def forward(self, input, target, keep_batch = False):
        if not keep_batch:
            mse = self.mse(input, target);
            norm_coeff = torch.var(target);
            return mse / norm_coeff;
        else:
            mse = self.mse_none(input, target).mean(dim = [1, 2, 3]);
            norm_coeff = torch.var(target, dim = [1, 2, 3]);
            return mse / norm_coeff;
    
def get_loss_func(name:str):
    if name.lower() == 'mse':
        return MSELoss();
    elif name.lower() == 'nmse':
        return NMSELoss();

def gaussian_kernel(x, y, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """

    """
    n_samples = int(x.size()[0]) + int(y.size()[0])
    total = torch.cat([x, y], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2) 

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val) / len(kernel_val)

def mmd_loss(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    Calculate the mmd loss
    """
    batch_size = int(source.size()[0])
    kernels = gaussian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]

    loss = torch.mean(XX + YY - XY - YX)
    return loss

def coral_loss(source, target):
    """
    Calculate CORAL Loss
    """
    d = source.size(1)
    n_s = source.size(0)
    n_t = target.size(0)
    
    source_c = (source - source.mean(dim=0)).t() @ (source - source.mean(dim=0)) / (n_s - 1)
    
    target_c = (target - target.mean(dim=0)).t() @ (target - target.mean(dim=0)) / (n_t - 1)
    
    loss = torch.norm(source_c - target_c, p='fro') ** 2
    
    loss = loss / (4 * d * d)
    
    return loss