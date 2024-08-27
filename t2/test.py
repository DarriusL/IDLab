import torch
import numpy as np
from prettytable import PrettyTable
from copy import deepcopy
from data.wrapper import data_wrapper
from lib import glb_var,json_util, decorator, colortext

logger = glb_var.get_value('logger');

class Tester():
    def __init__(self, config) -> None:
        save_dir = config['save_dir'];
        self.config = config
        self.best_model = torch.load(save_dir + 'model.pt')['model'];
        self.best_model.threshold_percents = config['model']['threshold_percents'];
        self.end_model = torch.load(save_dir + 'model_end.pt')['model'];
        self.end_model.threshold_percents = config['model']['threshold_percents'];

        self.test_loader, _ = data_wrapper(deepcopy(self.config), 'test');

        if self.best_model.name.lower() in ['csinet_dec']:
            self.test_func = self._ae_test;
        else:
            self.test_func = self._general_test;
    
    def _calculate(self, model, str_model, info = ''):
        ACC_NAME = 'Intrusion' if model.is_Intrusion_Detection  else 'Identification';
        test_acc_all = .0;
        test_acc = [];
        model.eval();
        with torch.no_grad():
            for amps, ids, envs in iter(self.test_loader):
                test_acc.append(model.cal_accuracy(amps, ids, envs));
        test_acc_all = np.mean(test_acc);
        logger.info(f'[{model.name}]-[test({info})]-[{str_model}]- \nAccuracy [{ACC_NAME}]:'+ colortext.RED + 
                    f'{test_acc_all * 100:.4f}%' + colortext.RESET);
        return test_acc_all;

    def _general_test(self):
        save_dir =  self.config['save_dir'];
        
        result = {'end model':{}, 'best model':{}};
        result['end model']['acc'] = self._calculate(self.end_model, 'end model');
        result['best model']['acc'] = self._calculate(self.best_model, 'best model');
        json_util.jsonsave(result, save_dir + 'result.json');

        from matplotlib import pyplot as plt
        import pickle
        with open(save_dir + 'ep.pkl', 'rb') as f:
            ep = pickle.load(f)

        with open(save_dir + 'fig_acc.pkl', 'rb') as f:
            fig_acc = pickle.load(f);
        plt.figure(fig_acc);
        plt.plot(ep, result['end model']['acc'] * np.ones_like(ep), label = f'test acc - end model');
        plt.plot(ep, result['best model']['acc'] * np.ones_like(ep), label = f'test acc - best model');
        plt.xlabel('epoch');
        plt.ylabel('loss');
        plt.legend(loc='upper right')
        plt.savefig(save_dir + 'test_acc.png', dpi = 400);

        logger.info(f'Saving dir:{save_dir}');
    
    def _ae_test(self):
        def _step(model, percent, str_model, info):
            model.set_threshold(percent);
            result[str_model].append(self._calculate(model, str_model, info));
        
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