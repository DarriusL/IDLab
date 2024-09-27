
from lib import glb_var
from model.framework.bird import BIRDEncoder, BIRD
from model.framework.caution.encoder import Encoder
from model.framework.caution.caution import Caution
from model.framework.gait_enhance import GaitEnhance
from model.framework.deep_wiid import DeepWiID
from model.framework.maiu import MAIUId, MAIU
from model.framework.csiid import CSIID
from model.framework.gateid import GateID
from model.framework.wiai_id import WiAiId
from model.framework.autoencoder import AE, CAE

device = glb_var.get_value('device');
logger = glb_var.get_value('logger');

__all__ = ['generate_model']

#TODO:simplify
def generate_model(model_cfg):
    if model_cfg['name'].lower() == 'bird':
        model = BIRD(model_cfg);
    elif model_cfg['name'].lower() == 'bird_encoder':
        model = BIRDEncoder(model_cfg);
    elif model_cfg['name'].lower() == 'caution_encoder':
        model = Encoder(model_cfg);
    elif model_cfg['name'].lower() == 'caution':
        model = Caution(model_cfg);
    elif model_cfg['name'].lower() == 'gait_enhance':
        model = GaitEnhance(model_cfg);
    elif model_cfg['name'].lower() == 'deep_wiid':
        model = DeepWiID(model_cfg);
    elif model_cfg['name'].lower() == 'maiu_id':
        model = MAIUId(model_cfg);
    elif model_cfg['name'].lower() == 'maiu':
        model = MAIU(model_cfg);
    elif model_cfg['name'].lower() == 'csiid':
        model = CSIID(model_cfg);
    elif model_cfg['name'].lower() == 'gait_id':
        model = GateID(model_cfg);
    elif model_cfg['name'].lower() == 'wiai_id':
        model = WiAiId(model_cfg);
    elif model_cfg['name'].lower() == 'ae':
        model = AE(model_cfg);
    elif model_cfg['name'].lower() == 'cae':
        model = CAE(model_cfg);
    else:
        raise NotImplementedError

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f'The number of parameters of the [{model.name}]: {num_params / 10 ** 6:.6f} M');
    return model.to(device);