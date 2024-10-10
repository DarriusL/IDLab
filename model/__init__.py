
import os
from lib import glb_var, util, json_util

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