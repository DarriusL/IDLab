# @Time   : 2023.07.08
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com

from typing import Any
from lib import glb_var, util
import time

logger = glb_var.get_value('logger');

class Decorator(object):
    '''Abstract Decorator class
    '''
    def __init__(self, func) -> None:
        self.func = func;

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        '''Method needs to be called after being implemented'''
        raise NotImplementedError;

class Timer(Decorator):
    '''Timer for func'''
    def __init__(self, func) -> None:
        super().__init__(func);

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        t = time.time();
        result = self.func(*args, **kwds);
        logger.info(f'The time consumption of {self.func.__name__}: {util.s2hms(time.time() - t)}');
        return result