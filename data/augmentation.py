import torch 
import numpy as np

def data_augmentation(batch, aug_cfg, device):
    def _augmentation(amps, ids, envs, s_aug_cfg, device):
        if s_aug_cfg['type'].lower() == 'gaussian_noise':
            amps_aug, ids_aug, envs_aug =  add_gaussian_noise(amps, ids, envs, s_aug_cfg, device);
        elif s_aug_cfg['type'].lower() == 'multipath_fading':
            amps_aug, ids_aug, envs_aug =  mutipath_fading(amps, ids, envs, s_aug_cfg, device);
        elif s_aug_cfg['type'].lower() == 'freq_selective_fading':
            amps_aug, ids_aug, envs_aug = freq_selective_fading(amps, ids, envs, s_aug_cfg, device);
        elif s_aug_cfg['type'].lower() == 'antenna_sequence':
            amps_aug, ids_aug, envs_aug = antenna_sequence(amps, ids, envs, s_aug_cfg, device);
        elif s_aug_cfg['type'].lower() == 'subcarrier_mask':
            amps_aug, ids_aug, envs_aug = subcarrier_mask(amps, ids, envs, s_aug_cfg, device);
        else:
            raise RuntimeError;

        return amps_aug, ids_aug, envs_aug
    amps, ids, envs = batch;
    new_amps, new_ids, new_envs = amps.clone(), ids.clone(), envs.clone();
    for s_aug_cfg in aug_cfg.values():
        amps_aug, ids_aug, envs_aug = _augmentation(amps.clone(), ids, envs, s_aug_cfg, device);
        new_amps = torch.cat((new_amps, amps_aug), dim = 0);
        new_ids = torch.cat((new_ids, ids_aug), dim = 0);
        new_envs = torch.cat((new_envs, envs_aug), dim = 0);
    return new_amps, new_ids, new_envs
        

def add_gaussian_noise(amps, ids, envs, s_aug_cfg, device):
    snr_db = s_aug_cfg['snr_db'];
    amps_n = amps + torch.normal(0, torch.sqrt(torch.mean(amps**2) / (10**(snr_db/10))), amps.shape, dtype = torch.float32, device = device);
    return amps_n, ids, envs;

def mutipath_fading(amps, ids, envs, s_aug_cfg, device):
    delay = s_aug_cfg['delay'];
    coef = s_aug_cfg['coef'];
    n = ids.shape[0];
    mp = np.random.choice(len(delay), n);
    amps_mp = amps.clone();
    for i in range(n):
        d = delay[mp[i]];
        c = coef[mp[i]];
        for j in range(len(d)):
            amps_mp[i, d[j]:, :] = amps_mp[i, d[j]:, :] + c[j] * amps[i, :-d[j], :];
    return amps_mp, ids, envs;

def freq_selective_fading(amps, ids, envs, s_aug_cfg, device):
    #amps:[N, T, R, F]
    num_f = amps.shape[-1];
    scopes = s_aug_cfg['scope'];
    scope = scopes[np.random.choice(len(scopes), 1, replace = False)[0]];
    coefs = torch.linspace(start = scope[0], end = scope[1], steps = num_f, device = device);
    return amps * coefs, ids, envs;

def antenna_sequence(amps, ids, envs, s_aug_cfg, device):
    #amps:[N, T, R, F]
    R = amps.shape[2];
    ant_idxs = np.random.choice(R, R, replace = False);
    amps_ant = amps[:, :, ant_idxs, :];
    return amps_ant, ids, envs;

def subcarrier_mask(amps, ids, envs, s_aug_cfg, device):
    #amps:[N, T, R, F]
    n_sc = int(s_aug_cfg['ratio'] * amps.shape[-1]);
    sc_idxs = np.random.choice(amps.shape[-1], n_sc, replace = False);
    amps[:, :, :, sc_idxs] = torch.zeros_like(amps[:, :, :, :n_sc]);
    return amps, ids, envs;
