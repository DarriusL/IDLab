import torch
from lib import glb_var, util

logger = glb_var.get_value('logger');

class Net(torch.nn.Module):
    '''Abstract Net class to define the API methods
    '''
    def __init__(self, model_cfg) -> None:
        super().__init__();
        util.set_attr(self, model_cfg, except_type = dict);
        self.is_Intrusion_Detection = False;

    def _init_para(self, module):
        if isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=1/module.embedding_dim)
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.fill_(1.0)
            module.weight.data.fill_(1.0)
        elif isinstance(module, torch.nn.Linear):
            module.weight.data.normal_()
            if module.bias is not None:
                module.bias.data.fill_(1.0)
    
    def forward(self):
        '''Defines the forward pass of the network. Must be implemented by subclasses.'''
        logger.error('Method needs to be called after being implemented');
        raise NotImplementedError;

    def encoder(self):
        '''Defines the encoder part of the network. Must be implemented by subclasses.'''
        logger.error('Method needs to be called after being implemented');
        raise NotImplementedError;

    @torch.no_grad()
    def cal_accuracy(self, amps, ids, envs) -> float:
        '''
        Calculates the accuracy of the model's predictions.
        
        If `is_Intrusion_Detection` is `False`, it calculates classification accuracy by comparing
        the predicted class with the ground truth (`ids`). If `is_Intrusion_Detection` is `True`,
        it calculates accuracy for intrusion detection tasks.

        Returns:
        ---------
        acc : float 
            The accuracy of the model's predictions.
        '''
        #ids:[N]
        #envs:[N]
        if not self.is_Intrusion_Detection:
            #p
            id_pred = self.p_classify(amps).argmax(dim = -1);
            acc = (id_pred == ids).cpu().float().mean().item();
        else:
            intrude_pred = self.intrusion_detection(amps);
            intrude_gt = ids >= self.known_p_num;
            acc = (intrude_gt == intrude_pred).cpu().float().mean().item();
        return acc;

    def p_classify(self, amps):
        '''Defines the id classiify part of the network. Must be implemented by subclasses.'''
        logger.error('Method needs to be called after being implemented');
        raise NotImplementedError;

    def train_epoch_hook(self, trainer, epoch):
        '''Hook for executing custom logic during each training epoch.'''
        pass

    def valid_epoch_hook(self, trainer, epoch):
        '''Hook for executing custom logic during each validation epoch.'''
        pass

    def pre_test_hook(self, tester):
        '''Hook for executing custom logic before testing the model.'''
        pass

    def cal_loss(self, amps, ids, envs, amps_t = None, ids_t = None, envs_t = None):
        '''Calculate the loss'''
        logger.error('Method needs to be called after being implemented');
        raise NotImplementedError;

    def conventional_train(self, X, Y):
        '''Only applicable to traditional methods'''
        logger.error('Method needs to be called after being implemented');
        raise NotImplementedError;

    def save(self, save_dir):
        '''
        General model saving method for pytorch model

        Saves the model's state dictionary to the specified directory.
        
        The model's parameters are saved as `model_state_dict.pth`.
        '''
        torch.save(self.state_dict(), save_dir + '/model_state_dict.pth');
        logger.info(f"Model saved to {save_dir}\n");

    def load(self, load_dir):
        '''General method for loading models

        Loads the model's state dictionary from the specified directory.
        
        The model's parameters are loaded from `model_state_dict.pth`.
        '''
        self.load_state_dict(torch.load(load_dir + '/model_state_dict.pth', weights_only = True));
        logger.info(f"Model loaded from {load_dir}\n")