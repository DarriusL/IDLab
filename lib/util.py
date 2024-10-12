# @Time   : 2022.03.03
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
import os, importlib, platform, subprocess, signal
import pydash as ps
import datetime, time, inspect
from matplotlib import pyplot as plt
from typing import Callable


def set_attr(obj, dict_, keys = None, except_type = None):
    '''Quickly assign properties to objects

    Parameters:
    -----------
    obj: object of a python class
        Assign values to the properties of the object

    dict_: dict
        A dictionary for assigning values to object attributes, 
        whose key is the attribute, and the corresponding value is the value
    
    keys: list, optional
        defalue: None
        Select the corresponding key in the dictionary to assign a value to the object

    '''
    if keys is not None:
        dict_ = ps.pick(dict, keys);
    for key, value in dict_.items():
        if type(value) == except_type:
            continue;
        setattr(obj, key, value);

def get_attr(obj, keys):
    '''Quick access to object properties

    Parameters:
    -----------

    obj: object of a python class
        Get the properties of the object
    
    keys: list
        A list of properties of the object that require
    
    Returns:
    --------

    dict: dict
        The required attribute is the value corresponding to the keys of the dictionary
    

    '''
    dict = dict();
    for key in keys:
        if hasattr(obj, key):
            dict[key] = getattr(obj, key);
    return dict;

def get_func_rets(func:Callable, params:dict):
    '''Get the output of a function

    Parameters:
    -----------
    func:function

    cfg:dict
    '''
    from lib import glb_var

    logger = glb_var.get_value('logger');
    parm_keys = tuple(inspect.signature(func).parameters.keys());
    ext_params = ps.pick(params, parm_keys);
    try:
        rets = func(**ext_params);
        logger.debug(f'Extracted parameters: {parm_keys}');
        logger.debug(f'Extracted names of parameter: {ext_params.keys()}');
        return rets;
    except:
        logger.error(f'Extracted parameters: {parm_keys}');
        logger.error(f'Extracted names of parameter: {ext_params.keys()}');
        logger.error(f'Extracted value of parameter: {ext_params.values()}');
        raise RuntimeError

def set_seed(seed):
    '''Set random seed
    '''
    import torch
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 

def get_date(separator = '_'):
    '''Quickly get formated date
    '''
    return str(datetime.date.today()).replace('-', separator);

def get_time(separator = '_'):
    '''Quickly get formated time
    '''
    now = datetime.datetime.now();
    t = f'{now.hour}-{now.minute}-{now.second}';
    return t.replace('-', separator);

def s2hms(s):
    '''Convert s to hms
    '''
    days = s // 86400;
    return f'{days} day(s) -' + time.strftime("%H hour - %M min - %S s", time.gmtime(s));

def load_modules_from_directory(directory, package):
    '''
    '''
    for filename in os.listdir(directory):
        if filename.endswith(".py") and filename != "__init__.py":
            module_name = filename[:-3]
            importlib.import_module(f"{package}.{module_name}")

def is_wsl():
    if platform.system().lower() == 'linux':
        with open('/proc/version', 'r') as f:
            version_info = f.read().lower()
            return 'microsoft' in version_info
    return False

def kill_process_on_port(port):
    from lib import glb_var, colortext
    logger = glb_var.get_value('logger');
    try:
        if platform.system().lower() == 'linux':
            # Linux: lsof
            logger.info(f"Checking for process on port {port} (Linux)");
            result = subprocess.check_output(f"lsof -i:{port} -t", shell=True);
            pid = int(result.strip());
            logger.info(colortext.YELLOW + f"Terminating process {pid}, which is occupying port {port}" + colortext.RESET);
            os.kill(pid, signal.SIGTERM);
        elif platform.system().lower() == 'windows':
            # Windows: netstat
            logger.info(f"Checking for process on port {port} (Windows)");
            result = subprocess.check_output(f"netstat -ano | findstr :{port}", shell=True);
            lines = result.decode().strip().split("\n");
            for line in lines:
                pid = int(line.strip().split()[-1]);
                logger.info(colortext.YELLOW + f"Terminating process {pid}, which is occupying port {port}" + colortext.RESET);
                subprocess.call(f"taskkill /F /PID {pid}", shell=True);
        else:
            logger.error(f"Unsupported platform: {platform.system()}");
    except subprocess.CalledProcessError:
        logger.info(f"No process is occupying port {port}");
    except Exception as e:
        logger.error(f"An error occurred while trying to kill the process on port {port}: {e}");