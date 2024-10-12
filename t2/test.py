import torch, os, tqdm, sys
import numpy as np
from prettytable import PrettyTable
from copy import deepcopy
from data.wrapper import data_wrapper
from lib import glb_var,json_util, decorator, colortext, callback
from model import load_model, generate_model

logger = glb_var.get_value('logger');

class Tester():
    def __init__(self, config) -> None:
        save_dir = config['save_dir'];
        self.config = config
        config['model']['known_p_num'] = config['known_p_num'];
        config['model']['known_env_num'] = config['known_env_num'];
        if config['model']['name'] == 'Caution':
            #Caution models do not require training
            self.best_model = generate_model(config['model']);
            self.end_model = self.best_model;
        else:
            self.best_model = load_model(save_dir + 'best/');
            self.end_model = load_model(save_dir + 'end/');

        self.test_loader, _ = data_wrapper(deepcopy(self.config), 'test');

        if self.best_model.name in ['CautionEncoder', 'Caution']:
            self.support_loader, _ = data_wrapper(deepcopy(self.config), 'train');
        if self.best_model.name in ['Caution']:
            self.valid_loader, _ = data_wrapper(deepcopy(self.config), 'valid');

        if self.best_model.name in ['BIRD', 'AE', 'CAE']:
            self.test_func = self._ae_test;
            self.best_model.threshold_percents = config['model']['threshold_percents'];
            self.end_model.threshold_percents = config['model']['threshold_percents'];
        else:
            self.test_func = self._general_test;
    
    def _calculate(self, model, str_model, info = ''):
        ACC_NAME = 'Intrusion' if model.is_Intrusion_Detection  else 'Identification';
        test_acc_all = .0;
        test_acc = [];
        model.eval();
        with torch.no_grad():
            for amps, ids, envs in tqdm.tqdm(self.test_loader, desc=f"[Test progress]", unit="batch", leave=False, file=sys.stdout):
                with callback.no_stdout():
                    test_acc.append(model.cal_accuracy(amps, ids, envs));
        test_acc_all = np.mean(test_acc);
        logger.info(f'[{model.name}]-[test({info})]-[{str_model}]- \nAccuracy [{ACC_NAME}]:'+ colortext.RED + 
                    f'{test_acc_all * 100:.4f}%\n' + colortext.RESET);
        return test_acc_all;

    def _general_test(self):
        self.best_model.pre_test_hook(self);
        self.end_model.pre_test_hook(self);

        save_dir =  self.config['save_dir'];
        
        result = {'end model':{}, 'best model':{}};
        result['end model']['acc'] = self._calculate(self.end_model, 'end model');
        result['best model']['acc'] = self._calculate(self.best_model, 'best model');
        json_util.jsonsave(result, save_dir + 'result.json');

        logger.info(f'Saving dir:{save_dir}');
    
    def _ae_test(self):
        def _step(model, percent, str_model, info):
            model.set_threshold(percent);
            result[str_model].append(self._calculate(model, str_model, info));
        self.best_model.pre_test_hook(self);
        self.end_model.pre_test_hook(self);
        logger.info('Load train loader to update threshold');
        train_loader, _ = data_wrapper(deepcopy(self.config), 'train');
        self.best_model.update_thresholds(train_loader);
        self.end_model.update_thresholds(train_loader);
        del train_loader;
    
        save_dir =  self.config['save_dir'];
        result = {'end model':[], 'best model':[]};
    
        for idx in range(len(self.best_model.threshold_percents)):
            percent = self.best_model.threshold_percents[idx];
            info = f'{percent}%-threshold';
            _step(self.best_model, percent, 'best model', info);
            _step(self.end_model, percent, 'end model', info);

        #show result
        table = PrettyTable();
        table.add_column('threshold_percent', self.best_model.threshold_percents);
        table.add_column('best model', result['best model']);
        table.add_column('end model', result['end model']);
        for row in table._rows:
            row[1] = f"{row[1] * 100:.5f} ";
            row[2] = f"{row[2] * 100:.5f} ";
            if float(row[1]) > float(row[2]):
                row[1] = f"{colortext.GREEN}{row[1]}%{colortext.RESET}";
                row[2] = f"{row[2]}%";
            elif float(row[2]) > float(row[1]):
                row[1] = f"{row[1]}%";
                row[2] = f"{colortext.GREEN}{row[2]}%{colortext.RESET}";
            else:
                row[1] = f"{colortext.GREEN}{row[1]}%{colortext.RESET}";
                row[2] = f"{colortext.GREEN}{row[2]}%{colortext.RESET}";
        title = 'Summary of Test Accuracy';
        title = title.center(len(table.get_string().splitlines()[0]));
        logger.info(f'{title}\n{table}');

        result_to_save = {'end model':{}, 'best model':{}};
        for p, acc_b, acc_e in zip(self.best_model.threshold_percents, result['best model'], result['end model']):
            result_to_save['best model'][f'acc_{p}'] = acc_b;
            result_to_save['end model'][f'acc_{p}'] = acc_e;
        json_util.jsonsave(result_to_save, save_dir + 'result.json');
        logger.info(f'Saving dir:{save_dir}');
    
    def test(self):
        self.test_func();

@decorator.Timer
def test_model(config:dict):
    tester = Tester(config);
    tester.test();