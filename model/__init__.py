
import os
from lib import glb_var, util, json_util
# from model.framework.bird import BIRDEncoder, BIRD, BIRDEncoderSVM
# from model.framework.caution import Caution, CautionEncoder
# from model.framework.caution import Caution
# from model.framework.gait_enhance import GaitEnhance
# from model.framework.deep_wiid import DeepWiID
# from model.framework.maiu import MAIUId, MAIU
# from model.framework.csiid import CSIID
# from model.framework.gateid import GateID
# from model.framework.wiai_id import WiAiId
# from model.framework.autoencoder import AE, CAE

device = glb_var.get_value('device');
logger = glb_var.get_value('logger');

__all__ = ['generate_model', 'load_model'];

framework_dir = os.path.join(os.path.dirname(__file__), 'framework')
util.load_modules_from_directory(framework_dir, 'model.framework')

def generate_model(model_cfg):
    model_class = glb_var.query_model(model_cfg['name'])
    if model_class is None:
        raise RuntimeError(f"Unable to find model class named {model_cfg['name']}, please check configuration")
    
    try:
        model = model_class(model_cfg);
    except Exception as e:
        raise RuntimeError(f"Error instantiating model {model_cfg['name']} : {e}")
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f'The number of parameters of the [{model.name}]: {num_params / 10 ** 6:.6f} M\n');
    return model.to(device);

def load_model(load_dir):
    '''
    if A/B/model.pth then load_dir = A/B/
    '''
    if os.path.isfile(load_dir):
        raise RuntimeError('Please use the directory of the file instead of its path');
    cfg = json_util.jsonload(load_dir + '/config.json');
    model = generate_model(cfg['model']);
    model.load(load_dir);
    return model.to(device);
    

#TODO:simplify
# def generate_model(model_cfg):
#     if model_cfg['name'].lower() == 'bird':
#         model = BIRD(model_cfg);
#     elif model_cfg['name'].lower() == 'bird_encoder':
#         model = BIRDEncoder(model_cfg);
#     elif model_cfg['name'].lower() == 'caution_encoder':
#         model = CautionEncoder(model_cfg);
#     elif model_cfg['name'].lower() == 'bird_encoder_svm':
#         model = BIRDEncoderSVM(model_cfg);
#     elif model_cfg['name'].lower() == 'caution':
#         model = Caution(model_cfg);
#     elif model_cfg['name'].lower() == 'gait_enhance':
#         model = GaitEnhance(model_cfg);
#     elif model_cfg['name'].lower() == 'deep_wiid':
#         model = DeepWiID(model_cfg);
#     elif model_cfg['name'].lower() == 'maiu_id':
#         model = MAIUId(model_cfg);
#     elif model_cfg['name'].lower() == 'maiu':
#         model = MAIU(model_cfg);
#     elif model_cfg['name'].lower() == 'csiid':
#         model = CSIID(model_cfg);
#     elif model_cfg['name'].lower() == 'gait_id':
#         model = GateID(model_cfg);
#     elif model_cfg['name'].lower() == 'wiai_id':
#         model = WiAiId(model_cfg);
#     elif model_cfg['name'].lower() == 'ae':
#         model = AE(model_cfg);
#     elif model_cfg['name'].lower() == 'cae':
#         model = CAE(model_cfg);
#     else:
#         raise NotImplementedError

#     num_params = sum(p.numel() for p in model.parameters())
#     logger.info(f'The number of parameters of the [{model.name}]: {num_params / 10 ** 6:.6f} M\n');
#     return model.to(device);