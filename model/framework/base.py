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
        '''network forward pass'''
        logger.error('Method needs to be called after being implemented');
        raise NotImplementedError;

    @torch.no_grad()
    def cal_accuracy(self, amps, ids, envs):
        #ids:[N]
        #envs:[N]
        if not self.is_Intrusion_Detection:
            #p
            feature = self.encoder(amps);
            id_pred = self.p_classifier(feature).argmax(dim = -1);
            acc = (id_pred == ids).cpu().float().mean().item();
        else:
            intrude_pred = self.intrusion_detection(amps);
            intrude_gt = ids >= self.known_p_num;
            acc = (intrude_gt == intrude_pred).cpu().float().mean().item();
        return acc;

    def after_train_hook(self, trainer):
        pass;

    def cal_loss(self):
        '''Calculate the loss'''
        logger.error('Method needs to be called after being implemented');
        raise NotImplementedError;