
from lib import glb_var
from model.framework.encoder.double_trans import DoubleTrans
from model.framework.encoder.aaconv_trans import AAconvTrans
from model.framework.encoder.csinet_trans import CSINetTrans
from model.framework.decoder.csinet_dec import CSINetDec
from model.framework.caution.encoder import Encoder
from model.framework.caution.caution import Caution
from model.framework.gait_enhance import GaitEnhance
from model.framework.deep_wiid import DeepWiID
from model.framework.maiu import MAIU
from model.framework.csiid import CSIID
from model.framework.gateid import GateID

device = glb_var.get_value('device');
logger = glb_var.get_value('logger');

__all__ = ['generate_model']

#TODO:simplify
def generate_model(model_cfg):
    if model_cfg['name'].lower() == 'double_trans':
        model = DoubleTrans(model_cfg);
    elif model_cfg['name'].lower() == 'aaconv_trans':
        model = AAconvTrans(model_cfg);
    elif model_cfg['name'].lower() == 'csinet_trans':
        model = CSINetTrans(model_cfg);
    elif model_cfg['name'].lower() == 'csinet_dec':
        model = CSINetDec(model_cfg);
    elif model_cfg['name'].lower() == 'caution_encoder':
        model = Encoder(model_cfg);
    elif model_cfg['name'].lower() == 'caution':
        model = Caution(model_cfg);
    elif model_cfg['name'].lower() == 'gait_enhance':
        model = GaitEnhance(model_cfg);
    elif model_cfg['name'].lower() == 'deep_wiid':
        model = DeepWiID(model_cfg);
    elif model_cfg['name'].lower() == 'maiu':
        model = MAIU(model_cfg);
    elif model_cfg['name'].lower() == 'csiid':
        model = CSIID(model_cfg);
    elif model_cfg['name'].lower() == 'gait_id':
        model = GateID(model_cfg);
    else:
        raise NotImplementedError

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f'The number of parameters of the [{model.name}]: {num_params / 10 ** 6:.6f} M');
    return model.to(device);