from lib import util, glb_var, decorator, json_util, callback, colortext
from copy import deepcopy
from data.wrapper import data_wrapper
from model import generate_model
from model.net_util import GradualWarmupScheduler
import numpy as np
import torch

logger = glb_var.get_value('logger');
device = glb_var.get_value('device');

class Trainer():
    def __init__(self, config) -> None:
        util.set_attr(self, config['train']);
        self.train_loader, _ = data_wrapper(deepcopy(config), 'train');
        self.valid_loader, info = data_wrapper(deepcopy(config), 'valid');

        #[N, T, R, F]
        config['model']['dim_in'] = info;

        config['model']['known_p_num'] = config['known_p_num'];
        config['model']['known_env_num'] = config['known_env_num'];
        self.config = config;

        model = generate_model(deepcopy(config['model']));
        self.model = model;
        self.optimizer = torch.optim.AdamW(
            params = filter(lambda p: p.requires_grad, model.parameters()),
            lr = config['train']["optimizer"]['lr'],
            weight_decay = config['train']["optimizer"]['weight_decay']
        );
        self.cosinSchduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer = self.optimizer,
            T_max = config["train"]['max_epoch'],
            eta_min = 0,
            last_epoch = -1
        );
        self.warmupScheduler = GradualWarmupScheduler(
            optimizer =  self.optimizer,
            multiplier = 2,
            warm_epoch = config["train"]['max_epoch'] // 10,
            after_scheduler = self.cosinSchduler
        );
    
    def _check_nan(self, loss):
        if torch.isnan(loss):
            logger.error('Loss is nan.');
            raise callback.CustomException('ValueError');

    def _train_step(self):
        epoch_train_loss, epoch_train_acc = [], [];
        self.model.train();
        for amps, ids, envs in iter(self.train_loader):
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache();
            self.optimizer.zero_grad();
            loss = self.model.cal_loss(amps, ids, envs);
            self._check_nan(loss);
            loss.backward();
            torch.nn.utils.clip_grad_norm_(
                parameters = self.model.parameters(),
                max_norm = 1
            );
            self.optimizer.step();
            epoch_train_loss.append(loss.cpu().item());
            epoch_train_acc.append(self.model.cal_accuracy(amps, ids, envs));
        return np.mean(epoch_train_loss), np.mean(epoch_train_acc);

    def train(self):
        ACC_NAME = 'Intrusion' if self.model.is_Intrusion_Detection  else 'Identification';
        save_dir = self.config['save_dir'];
        train_loss, valid_loss = [], [];
        train_acc, valid_acc = [], [];
        valid_best = np.inf if self.valid_metrics_less else -np.inf;
        valid_cur = None;
        valid_cnt = 0;
        for epoch in range(self.max_epoch):
            epoch_train_loss, accuracy = self._train_step();
            train_loss.append(epoch_train_loss);
            train_acc.append(accuracy);
            acc_info = '' if self.valid_metrics.lower() == 'loss' else f'Accuracy {ACC_NAME}: {train_acc[-1] * 100 :.6f} %';
            logger.info(f'[{self.model.name}]-[train]-[{device}] - [{self.config["data"]["dataset"]}] - [metrics: {self.valid_metrics.lower()}]\n'
                        f'[epoch: {epoch + 1}/{self.max_epoch}] - lr:{self.warmupScheduler.get_last_lr()[0]:.8f}\n' 
                        f'train loss:{train_loss[-1]:.8f}\n'+
                        acc_info);
            self.warmupScheduler.step();

            if (epoch + 1) >= self.valid_start_epoch and (epoch + 1) % self.valid_step == 0:
                epoch_valid_loss, epoch_valid_acc = [], [];
                self.model.eval();
                with torch.no_grad():
                    for amps, ids, envs in iter(self.valid_loader):
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache();
                        epoch_valid_loss.append(self.model.cal_loss(amps, ids, envs, is_target_data = self.is_DA).cpu().item());
                        epoch_valid_acc.append(self.model.cal_accuracy(amps, ids, envs));
                valid_loss.append(np.mean(epoch_valid_loss));
                valid_acc.append(np.mean(epoch_valid_acc));
                if self.valid_metrics.lower() == 'loss':
                    valid_cur = valid_loss[-1];
                elif self.valid_metrics.lower() == 'acc':
                    valid_cur = valid_acc[-1];
                else:
                    raise RuntimeError;
            
                comparison = (valid_cur <= valid_best) if self.valid_metrics_less else (valid_cur >= valid_best);
                if comparison:
                    valid_best = valid_cur;
                    valid_cnt = 0;
                    torch.save({'model':self.model, 'epoch':epoch, 'config':self.config}, save_dir + 'model.pt');
                else:
                    valid_cnt += 1;
                acc_info = '' if self.valid_metrics.lower() == 'loss' else f'Accuracy {ACC_NAME}: {valid_acc[-1] * 100 : .6f} % / {np.max(valid_acc) * 100: .6f} % (best)';
                logger.info(f'[{self.model.name}]-['+ colortext.PURPLE +'valid' + colortext.RESET +f']-[metrics: {self.valid_metrics.lower()}]\n'
                    f'[epoch: {epoch + 1}/{self.max_epoch}] \n'
                    f'valid loss:{valid_loss[-1]:.8f} / {np.min(valid_loss):.8f} (best)' +
                    acc_info + '\n' +
                    colortext.GREEN + f'valid_not_imporove_cnt: {valid_cnt}' + colortext.RESET);
                if valid_cnt >= self.stop_train_step_valid_not_improve:
                    logger.info('Reaching the set stop condition')
                    break;

        torch.save({'model':self.model, 'epoch':epoch}, save_dir + 'model_end.pt');
        json_util.jsonsave(self.config, save_dir + 'config.json');
        self.model.after_train_hook(self);
        
        from matplotlib import pyplot as plt
        import pickle
        fig_loss = plt.figure(figsize = (10, 6));
        ep = np.arange(0, len(train_loss)) + 1;
        #train loss
        plt.plot(ep, train_loss, label = 'train loss');
        #valid loss
        plt.plot(np.arange(self.valid_start_epoch - 1, len(train_loss), self.valid_step) + 1, valid_loss, label = 'valid loss');
        plt.xlabel('epoch');
        plt.ylabel('loss');
        plt.yscale('log');
        plt.legend(loc='upper right')
        plt.savefig(save_dir + 'loss.png', dpi = 400); 
        with open(save_dir + 'ep.pkl', 'wb') as f:
            pickle.dump(ep, f);
        with open(save_dir + 'fig_loss.pkl', 'wb') as f:
            pickle.dump(fig_loss, f);

        fig_acc = plt.figure(figsize = (10, 6));
        #train acc
        plt.plot(ep, train_acc, label = 'train acc');
        #valid acc
        plt.plot(np.arange(self.valid_start_epoch - 1, len(train_acc), self.valid_step) + 1, valid_acc, label = 'valid acc');
        plt.xlabel('epoch');
        plt.ylabel('Accuracy');
        plt.legend(loc='upper right');
        plt.savefig(save_dir + 'acc.png', dpi = 400); 
        with open(save_dir + 'fig_acc.pkl', 'wb') as f:
            pickle.dump(fig_acc, f);

class DATrainer(Trainer):
    def __init__(self, config) -> None:
        super().__init__(config)
        cfg_t = deepcopy(config);
        if 'augmentation' in config['data']['train'].keys():
            cfg_t['data']['target']['augmentation'] = cfg_t['data']['train']['augmentation'];
        self.target_loader, _ = data_wrapper(cfg_t, 'target');

    def _train_step(self):
        epoch_train_loss, epoch_train_acc = [], [];
        self.model.train();
        for amps, ids, envs in iter(self.train_loader):
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache();
            self.optimizer.zero_grad();
            train_loss = self.model.cal_loss(amps, ids, envs);
            amps_t, ids_t, envs_t = iter(self.target_loader).__next__();
            target_loss = self.model.cal_loss(amps_t, ids_t, envs_t, is_target_data = True); 
            loss = train_loss + target_loss;
            self._check_nan(loss);
            loss.backward();
            torch.nn.utils.clip_grad_norm_(
                parameters = self.model.parameters(),
                max_norm = 1
            );
            self.optimizer.step();
            epoch_train_loss.append(loss.cpu().item());
            epoch_train_acc.append(self.model.cal_accuracy(amps, ids, envs));
            epoch_train_acc.append(self.model.cal_accuracy(amps_t, ids_t, envs_t))
        return np.mean(epoch_train_loss), np.mean(epoch_train_acc);

class CautionTrainer(Trainer):
    def __init__(self, config) -> None:
        super().__init__(config);
        self.support_loader = self.train_loader;
        self.query_loader = self.valid_loader;
    
    def _train_step(self, epoch):
        epoch_train_loss, epoch_train_acc = [], [];
        if ((epoch + 1) >= self.update_start_epoch and (epoch + 1) % self.update_step == 0) or epoch  == 0:
            self.model.update_center(self.support_loader);
            logger.info('Caution: update!');
        self.model.train();
        for amps, ids, envs in iter(self.query_loader):
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache();
            self.optimizer.zero_grad();
            loss = self.model.cal_loss(amps, ids, envs);
            self._check_nan(loss);
            loss.backward();
            torch.nn.utils.clip_grad_norm_(
                parameters = self.model.parameters(),
                max_norm = 1
            );
            self.optimizer.step();
            epoch_train_loss.append(loss.cpu().item());
            epoch_train_acc.append(self.model.cal_accuracy(amps, ids, envs));
        return np.mean(epoch_train_loss), np.mean(epoch_train_acc);


    def train(self):
        if self.model.is_Intrusion_Detection: return;
        ACC_NAME = 'Identification';
        save_dir = self.config['save_dir'];
        train_loss, train_acc= [], [];
        acc_best = .0;
        for epoch in range(self.max_epoch):
            epoch_train_loss, accuracy = self._train_step(epoch);
            train_loss.append(epoch_train_loss);
            train_acc.append(accuracy);
            acc_info = '' if self.valid_metrics.lower() == 'loss' else f'Accuracy {ACC_NAME}: {train_acc[-1] * 100 :.6f} %';
            logger.info(f'[{self.model.name}]-[train]-[{device}] - [{self.config["data"]["dataset"]}] - [metrics: {self.valid_metrics.lower()}]\n'
                        f'[epoch: {epoch + 1}/{self.max_epoch}] - lr:{self.warmupScheduler.get_last_lr()[0]:.8f}\n' 
                        f'train loss:{train_loss[-1]:.8f}\n'+
                        acc_info);
            self.warmupScheduler.step();
            if train_acc[-1] >= acc_best:
                acc_best = train_acc[-1];
                torch.save({'model':self.model, 'epoch':epoch}, save_dir + 'model.pt');
        torch.save({'model':self.model, 'epoch':epoch}, save_dir + 'model_end.pt');
        json_util.jsonsave(self.config, save_dir + 'config.json');
        self.model.after_train_hook(self);
        
        from matplotlib import pyplot as plt
        import pickle
        fig_loss = plt.figure(figsize = (10, 6));
        ep = np.arange(0, len(train_loss)) + 1;
        #train loss
        plt.plot(ep, train_loss, label = 'train loss');

        plt.xlabel('epoch');
        plt.ylabel('loss');
        plt.yscale('log');
        plt.legend(loc='upper right')
        plt.savefig(save_dir + 'loss.png', dpi = 400); 
        with open(save_dir + 'ep.pkl', 'wb') as f:
            pickle.dump(ep, f);
        with open(save_dir + 'fig_loss.pkl', 'wb') as f:
            pickle.dump(fig_loss, f);

        fig_acc = plt.figure(figsize = (10, 6));
        #train acc
        plt.plot(ep, train_acc, label = 'train acc');

        plt.xlabel('epoch');
        plt.ylabel('Accuracy');
        plt.legend(loc='upper right');
        plt.savefig(save_dir + 'acc.png', dpi = 400); 
        with open(save_dir + 'fig_acc.pkl', 'wb') as f:
            pickle.dump(fig_acc, f);

        
def generate_trainer(config):
    if config['model']['name'].lower() in ['caution_encoder']:
        return CautionTrainer(config);
    if config['train']['is_DA']:
        trainer = DATrainer(config);
    else:
        trainer = Trainer(config);
    return trainer;

@decorator.Timer
def train_model(config):
    trainer = generate_trainer(config);
    trainer.train();
    return trainer.config;