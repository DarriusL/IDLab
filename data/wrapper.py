from lib import util, glb_var, decorator
import torch, h5py, platform
from data.augmentation import data_augmentation
from model import net_util
from copy import deepcopy

logger = glb_var.get_value('logger');
device = glb_var.get_value('device');

class generaldataset(torch.utils.data.Dataset):
    def __init__(self, attrs:dict) -> None:
        super().__init__();
        util.set_attr(self, attrs);
        if self.mode == 'v1':
            self.getitem_func = self.v1_getitem;
        elif self.mode == 'v3':
            self.getitem_func = self.v3_getitem;
        else:
            raise RuntimeError;
        

    def __len__(self) -> int:
        return self.n;

    def v1_getitem(self, index):
        return self.amps[index, :], self.ids[index], self.envs[index];

    def v3_getitem(self, index):
        return self.data[index]['amps'], self.data[index]['ids'], self.data[index]['envs'];

    def __getitem__(self, index):
        return self.getitem_func(index);

class loader_wrapper():
    def __init__(self, loader, aug_cfg) -> None:
        self.loader = loader;
        self.acfg = deepcopy(aug_cfg);
        self.aug_cfg = None;
        self.enable_aug();
    
    def enable_aug(self):
        self.aug_cfg = self.acfg;
        if self.aug_cfg is not None:
            logger.info(f'Data augmentation has been enabled');
            if net_util.get_free_gpu_memory(device) >= 4:
                self.aug_device = device;
            else:
                self.aug_device = torch.device('cpu');
            logger.info(f'Data augmentation device: {self.aug_device}');
    
    def disable_aug(self):
        self.aug_cfg = None;
        logger.info(f'Data augmentation has been disabled');

    def __iter__(self):
        for batch in self.loader:
            if self.aug_cfg is not None:
                batch = (data.to(self.aug_device) for data in batch);
                batch = data_augmentation(batch, self.aug_cfg, self.aug_device);
            yield (data.to(device) for data in batch);

@decorator.Timer
def v1_wrapper(config:dict, mode):
    data = h5py.File(config['data'][mode]['dir'], 'r');
    data_dict = {};
    data_dict['mode'] = 'v1';
    data_dict['n'] = data['data']['env'].shape[0];
    logger.info(f'Length of {mode} data:{data_dict["n"]}');
    #[N, T, R, F] -> [N, 6000, 3, 56]
    data_dict['amps'] = torch.tensor(data['data']['amp'][:]).permute(3, 2, 1, 0).float();
    
    #[N]
    data_dict['ids'] = torch.tensor(data['data']['id'][:, 0], dtype = torch.int64);
    data_dict['envs'] = torch.tensor(data['data']['env'][:, 0], dtype = torch.int64);
    del data;

    loader_param = config['data'][mode]['loader'];
    if platform.system().lower() == 'linux':
        pass
    else:
        loader_param['num_workers'] = 0;
        loader_param['prefetch_factor'] = None;
    loader_param['dataset'] = generaldataset(data_dict);
    aug_cfg = config['data'][mode]['augmentation'] if 'augmentation' in config['data'][mode].keys() else None;
    loader = loader_wrapper(torch.utils.data.DataLoader(**loader_param), aug_cfg);
    info = data_dict['amps'][:config['data'][mode]['loader']['batch_size'], :].shape;
    return (loader, info)

@decorator.Timer
def v3_warapper(config:dict, mode):
    #TODO:if train, returns all; if valid, ...
    data = h5py.File(config['data'][mode]['dir'], 'r');
    data_dict = {};
    data_dict['mode'] = 'v3';
    data_dict['n'] = config['known_p_num'];
    data_dict['data'] = [];
    for p_id in range(config['known_p_num']):
        p_id_data = {}
        p_id_data['amps'] = torch.tensor(data['data'][f'id{p_id}']['amp'][:]).permute(3, 2, 1, 0);
        p_id_data['ids'] = torch.tensor(data['data'][f'id{p_id}']['id'][:, 0], dtype = torch.int64);
        p_id_data['envs'] = torch.tensor(data['data'][f'id{p_id}']['env'][:, 0], dtype = torch.int64);
        data_dict['data'].append(p_id_data);
    del data

    loader_param = config['data'][mode]['loader'];
    assert (loader_param['batch_size'] == 1);
    if platform.system().lower() == 'linux':
        pass
    else:
        loader_param['num_workers'] = 0;
        loader_param['prefetch_factor'] = None;
    loader_param['dataset'] = generaldataset(data_dict);
    aug_cfg = config['data'][mode]['augmentation'] if 'augmentation' in config['data'][mode].keys() else None;
    loader = loader_wrapper(torch.utils.data.DataLoader(**loader_param), aug_cfg);
    return (loader, None)



def data_wrapper(config:dict, mode):
    if config['data']['dataset'] in ['v1', 'v2']:
        rets = v1_wrapper(config, mode);
    elif config['data']['dataset'] in ['v3']:
        rets = v3_warapper(config, mode);
    else:
        raise RuntimeError
    
    return rets;

