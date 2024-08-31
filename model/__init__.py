
from lib import glb_var
from model.framework.encoder.double_trans import DoubleTrans
from model.framework.encoder.aaconv_trans import AAconvTrans
from model.framework.encoder.csinet_trans import CSINetTrans
from model.framework.decoder.csinet_dec import CSINetDec
from model.framework.caution.encoder import Encoder

device = glb_var.get_value('device');

__all__ = ['generate_model']

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
    else:
        raise NotImplementedError

    return model.to(device);