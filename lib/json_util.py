# @Time   : 2023.03.03
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
import json, os

def jsonload(src):
    '''Read the json file and convert it to dict
       Suitable for files containing only one json string.

    Parameters: 
    -----------

    src:chr 
    the source direction of the jsonfile

    Returns:
    --------

    json_dict: dict
    '''
    assert os.path.exists(src);
    f = open(src, 'r');
    return json.load(f);

def jsonparse(src):
    '''Read the json file and convert every json string to dict
       Applicable to files containing only multiple json strings

    Parameters:
    -----------

    src: chr
    the source direction of the jsonfile

    Returns:
    --------

    json_dict: dict

    Example:
    ---------

    >>> from json_util import jsonparse
    >>> src = './file.json';
    >>> for item in jsonparse(src):
    >>>     print(item);
    '''
    assert os.path.exists(src);
    f = open(src, 'r');
    for item in f:
        yield json.loads(item)
    
def jsonlen(src):
    '''Count the number of json data in the json type file

    Parameters:
    -----------

    src:chr
    the source direction of the jsonfile

    Returns:
    --------
    
    cnt:int

    '''
    assert os.path.exists(src);
    f = open(src, 'r');
    cnt = 0;
    for item in f:
        if type(json.loads(item)) == dict:
            cnt += 1;
    return cnt;

def dict2jsonstr(dict):
    '''turn dict to str
    '''
    return json.dumps(dict, indent = 4);

def jsonsave(dict, tgt):
    '''Save the python type dict as a json file

    Parameters:
    -----------
    dict:dict

    tgt:str
    Destination path (including file name)
    '''
    json_data = json.dumps(dict, indent = 4);
    json_file = open(tgt, 'w');
    json_file.write(json_data);
    json_file.close();
