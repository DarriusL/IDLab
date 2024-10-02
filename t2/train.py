from lib import util, glb_var, decorator, json_util, callback, colortext
from copy import deepcopy
from data.wrapper import data_wrapper
from model import generate_model
from model.net_util import GradualWarmupScheduler
import numpy as np
import torch, tqdm, sys, time, os, warnings

logger = glb_var.get_value('logger');
device = glb_var.get_value('device');
tb_writer = glb_var.get_value('tb_writer');

class DLBaseTrainer(object):
    def __init__(self, config) -> None:
        util.set_attr(self, config['train']);
        self.train_loader, _ = data_wrapper(deepcopy(config), 'train');
        self.valid_loader, info = data_wrapper(deepcopy(config), 'valid');
        #[N, T, R, F]
        config['model']['dim_in'] = info;

        config['model']['known_p_num'] = config['known_p_num'];
        config['model']['known_env_num'] = config['known_env_num'];
        self.config = config;
        self.save_dir = config['save_dir'];

        #generalte model
        model = generate_model(deepcopy(config['model']));
        self.model = model;
        self._init_optimizer();
        self._init_scheduler();

    def _init_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            params = filter(lambda p: p.requires_grad, self.model.parameters()),
            lr = self.config['train']["optimizer"]['lr'],
            weight_decay = self.config['train']["optimizer"]['weight_decay']
        );

    def _init_scheduler(self):
        cosinSchduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer = self.optimizer,
            T_max = self.max_epoch,
            eta_min = 0,
            last_epoch = -1
        );
        self.scheduler = GradualWarmupScheduler(
            optimizer =  self.optimizer,
            multiplier = 2,
            warm_epoch = self.config["train"]['max_epoch'] // 10,
            after_scheduler = cosinSchduler
        );

    def _check_nan(self, loss):
        if torch.isnan(loss):
            logger.error('Loss is nan.');
            raise callback.CustomException('ValueError');

    def _clear_cache(self):
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache();

    def _log_epoch_info(self, epoch, loss, acc, t_train_start, is_train, valid_cnt = None):
        ACC_NAME = 'Intrusion' if self.model.is_Intrusion_Detection  else 'Identification';
        acc_info = '' if self.valid_metrics.lower() == 'loss' else f'Accuracy {ACC_NAME}: {acc[-1] * 100 :.6f} %\n';
        time_info = f'Accumulated training time:[{util.s2hms(time.time() - t_train_start)}]\n' + \
                    'Estimated time remaining:[' + colortext.YELLOW + \
                    f'{util.s2hms((time.time() - t_train_start)/(epoch + 1) * (self.max_epoch - epoch - 1))}' + colortext.RESET + ']\n';
        if is_train:
            logger.info(f'[{self.model.name}]-[train]-[{device}] - [{self.config["data"]["dataset"]}] - [metrics: {self.valid_metrics.lower()}]\n'
                        f'[epoch: {epoch + 1}/{self.max_epoch}] - lr:{self.scheduler.get_last_lr()[0]:.8f}\n' 
                        f'train loss:{loss[-1]:.8f}\n'+
                        acc_info + time_info);
        else:#valid
            logger.info(f'[{self.model.name}]-['+ colortext.PURPLE +'valid' + colortext.RESET +f']-[metrics: {self.valid_metrics.lower()}]\n'
                    f'[epoch: {epoch + 1}/{self.max_epoch}] \n'
                    f'valid loss:{loss[-1]:.8f} / {np.min(loss):.8f} (best)\n' +
                    acc_info +
                    colortext.GREEN + f'valid_not_imporove_cnt: {valid_cnt}\n' + colortext.RESET + time_info);

    def _save(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir);
        self.model.save(save_dir);
        json_util.jsonsave(self.config, save_dir + 'config.json');
    
    def _epoch_hook(self, epoch):
        if self.model.training:
            self.model.train_epoch_hook(self, epoch);
        else:
            self.model.valid_epoch_hook(self, epoch);
    
    def _common_step(self, epoch, loader, is_train = True):
        self._epoch_hook(epoch);
        epoch_loss, epoch_acc = [], [];
        msg = 'train' if is_train else 'valid';
        for amps, ids, envs in tqdm.tqdm(loader, desc=f"[epoch {epoch + 1}/{self.max_epoch}]-[{msg}]", unit="batch", leave=False, file=sys.stdout):
            with callback.no_stdout():
                self._clear_cache();
                if is_train:
                    self.optimizer.zero_grad();
                loss = self.model.cal_loss(amps, ids, envs);
                self._check_nan(loss);
                if is_train:
                    loss.backward();
                    torch.nn.utils.clip_grad_norm_(
                        parameters = filter(lambda p: p.requires_grad, self.model.parameters()),
                        max_norm = 1
                    );
                    self.optimizer.step();
                epoch_loss.append(loss.cpu().item());
                epoch_acc.append(self.model.cal_accuracy(amps, ids, envs));
        return np.mean(epoch_loss), np.mean(epoch_acc);

    def _train_step(self, epoch):
        self.model.train();
        return self._common_step(epoch, self.train_loader, is_train = True);

    @torch.no_grad()
    def _valid_step(self, epoch):
        self.model.eval();
        return self._common_step(epoch, self.valid_loader, is_train = False);

    def _after_train_epoch_hook(self, epoch):
        pass

    def train(self):
        t_train_start = time.time();
        self.train_loss, self.valid_loss = [], [];
        self.train_acc, self.valid_acc = [], [];
        valid_best = np.inf if self.valid_metrics_less else -np.inf;
        valid_cur = None;
        valid_cnt = 0;
        for epoch in range(self.max_epoch):
            epoch_train_loss, epoch_train_acc = self._train_step(epoch);
            #TODO: Add check point of TensorBoard
            tb_writer.add_scalar('Train loss', epoch_train_loss, epoch);
            self.train_loss.append(epoch_train_loss);
            tb_writer.add_scalar('Train acc', epoch_train_acc, epoch);
            self.train_acc.append(epoch_train_acc);
            self._log_epoch_info(epoch, self.train_loss, self.train_acc, t_train_start, is_train = True);
            self.scheduler.step();
            self._after_train_epoch_hook(epoch);

            if (epoch + 1) >= self.valid_start_epoch and (epoch + 1) % self.valid_step == 0:
                self._clear_cache();
                epoch_valid_loss, epoch_valid_acc = self._valid_step(epoch);
                #TODO: Add check point of TensorBoard
                self.valid_loss.append(epoch_valid_loss);
                self.valid_acc.append(epoch_valid_acc);
                if self.valid_metrics.lower() == 'loss':
                    valid_cur = self.valid_loss[-1];
                    tb_writer.add_scalar('Valid loss', epoch_valid_loss, epoch);
                elif self.valid_metrics.lower() == 'acc':
                    valid_cur = self.valid_acc[-1];
                    tb_writer.add_scalar('Valid acc', epoch_valid_acc, epoch);
                else:
                    raise RuntimeError;
                comparison = (valid_cur <= valid_best) if self.valid_metrics_less else (valid_cur >= valid_best);
                if comparison:
                    valid_best = valid_cur;
                    valid_cnt = 0;
                    self._save(self.save_dir + 'best/');
                else:
                    valid_cnt += 1;
                self._log_epoch_info(epoch, self.valid_loss, self.valid_acc, t_train_start, is_train = False, valid_cnt = valid_cnt);
                if valid_cnt >= self.stop_train_step_valid_not_improve:
                    logger.info('Reaching the set stop condition')
                    break;
        self._save(self.save_dir + 'end/');
        json_util.jsonsave(self.config, self.save_dir + 'config.json');
        #no more plot

class DATrainer(DLBaseTrainer):
    def __init__(self, config) -> None:
        cfg_t = deepcopy(config);
        super().__init__(config);
        if 'augmentation' in config['data']['train'].keys():
            cfg_t['data']['target']['augmentation'] = cfg_t['data']['train']['augmentation'];
        self.target_loader, _ = data_wrapper(cfg_t, 'target');

    def _train_step(self, epoch):
        self.model.train();
        target_iter = iter(self.target_loader);
        epoch_loss, epoch_acc = [], [];
        for amps, ids, envs in tqdm.tqdm(self.train_loader, desc=f"[epoch {epoch + 1}/{self.max_epoch}]-[train]", unit="batch", leave=False, file=sys.stdout):
            with callback.no_stdout():
                self._clear_cache();
                self.optimizer.zero_grad();
                try:
                    amps_t, ids_t, envs_t = next(target_iter);
                except StopIteration:
                    target_iter = iter(self.target_loader);
                    amps_t, ids_t, envs_t = next(target_iter);
                loss = self.model.cal_loss(amps, ids, envs, amps_t, ids_t, envs_t);
                self._check_nan(loss);
                loss.backward();
                torch.nn.utils.clip_grad_norm_(
                    parameters = filter(lambda p: p.requires_grad, self.model.parameters()),
                    max_norm = 1
                );
                self.optimizer.step();
                epoch_loss.append(loss.cpu().item());
                epoch_acc.append(self.model.cal_accuracy(amps, ids, envs));
                epoch_acc.append(self.model.cal_accuracy(amps_t, ids_t, envs_t));
        return np.mean(epoch_loss), np.mean(epoch_acc);

class CautionEncTrainer(DLBaseTrainer):
    def __init__(self, config) -> None:
        super().__init__(config);
        #do not valid
        if self.max_epoch >= self.valid_start_epoch:
            self.valid_start_epoch = self.max_epoch + 10;
            warnings.warn(f'The CautionEncoder model does not require a validation phase, and valid_start_epoch has been set to {self.valid_start_epoch} to skip validation.');
        self.support_loader = self.train_loader;
        self.query_loader = self.valid_loader;

    def _train_step(self, epoch):
        self.model.train();
        return self._common_step(epoch, self.query_loader, is_train = True);

    def _after_train_epoch_hook(self, epoch):
        if epoch == 0 or self.train_acc[-1] >= np.max(self.train_acc[:-1]):
            self._save(self.save_dir + 'best/');
        
class ConventionalTrainer(object):
    def __init__(self, config) -> None:
        util.set_attr(self, config['train']);

        config['model']['known_p_num'] = config['known_p_num'];
        config['model']['known_env_num'] = config['known_env_num'];

        self.train_loader, _ = data_wrapper(deepcopy(config), 'train');
        if 'valid' in config['data'].keys():
            self.valid_loader, _ = data_wrapper(deepcopy(config), 'valid');
        model = generate_model(deepcopy(config['model']));
        self.model = model;
        self.save_dir = config['save_dir'];
        self.config = config;
    
    def _save(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir);
        self.model.save(save_dir);
        json_util.jsonsave(self.config, save_dir + 'config.json');

    @torch.no_grad()
    def train(self):
        X = torch.zeros((0, self.model.do), device = device);
        Y = torch.zeros((0), device = device, dtype = torch.int64);
        for amps, ids, envs in tqdm.tqdm(self.train_loader, desc=f"Generating training data for {self.model.name}", unit="batch", leave=False, file=sys.stdout):
            with callback.no_stdout():
                X = torch.cat((
                    X, 
                    self.model.encoder(amps)
                    ), dim = 0);
                #if is Intrusion Detection, 1 for illegel, 0 for legal
                _Y = ids if not self.model.is_Intrusion_Detection else (ids >= self.model.known_p_num).to(torch.int64);
                Y = torch.cat((Y, _Y), dim = 0);
        X = X.cpu().detach().numpy();
        Y = Y.cpu().detach().numpy();
        self.model.conventional_train(X, Y);
        self._save(self.save_dir + 'best/');
        self._save(self.save_dir + 'end/');
        json_util.jsonsave(self.config, self.save_dir + 'config.json');

def generate_trainer(config):
    if config['model']['name'] == 'CautionEncoder':
        return CautionEncTrainer(config);
    elif config['model']['name'] in ['Caution']:
        ##Caution models do not require training
        return None;
    elif config['model']['name'] in ['BIRDEncoderSVM']:
        return ConventionalTrainer(config);

    if config['train']['is_DA']:
        trainer = DATrainer(config);
    else:
        trainer = DLBaseTrainer(config);
    return trainer;

@decorator.Timer
def train_model(config):
    trainer = generate_trainer(config);
    if trainer is None:
        return config;
    trainer.train();
    return trainer.config;