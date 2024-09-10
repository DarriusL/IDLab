import argparse, logging, os, torch, platform
from lib import glb_var, json_util, util

if __name__ == '__main__':
    glb_var.__init__();
    parse = argparse.ArgumentParser();
    parse.add_argument('--config', '-cfg', type = str, default = None, help = 'config for run');
    parse.add_argument('--saved_config', '-sc', type = str, default = None, help = 'path for saved config to test')
    parse.add_argument('--auto_shutdown', '-as', type = bool, default = False, help = 'automatic shutdown after program completion')

    args = parse.parse_args();
    is_train, is_test = False, False;
    if args.config is not None:
        config = json_util.jsonload(args.config);
        is_train = True;
        is_test = True;
    elif args.saved_config is not None:
        config = json_util.jsonload(args.saved_config);
        save_dir = os.path.dirname(args.saved_config) + '/';
        config['save_dir'] = save_dir;
        is_test = True;

    from lib.callback import Logger
    DATASET = config['data']['dataset'];
    INFO_LEVEL = logging.INFO;
    MODEL_NAME = config['model']['name'];

    if not os.path.exists(f'./cache/logger/{DATASET}/'):
        os.makedirs(f'./cache/logger/{DATASET}/');
    if 'save_dir' not in locals():
        save_dir = f'./cache/save/{DATASET}/{MODEL_NAME}_{util.get_date()}_{util.get_time()}/';
        config['save_dir'] = save_dir;
    if not os.path.exists(save_dir):
        os.makedirs(save_dir);
    glb_var.__init__();
    logger = Logger(
        level = INFO_LEVEL,
        filename = f'./cache/logger/{DATASET}/{MODEL_NAME}_{util.get_date()}_{util.get_time()}.log',
    ).get_log();
    logger.debug(f'save dir:{save_dir}');
    glb_var.set_value('logger', logger);
    if config['gpu_is_available']:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu');
    else:
        device = torch.device('cpu');
    glb_var.set_value('device', device);
    util.set_seed(config['seed']);

    from t2 import train, test
    if is_train:
        config = train.train_model(config);
    if is_test:
        test.test_model(config);

    if args.auto_shutdown:
        logger.info('Automatic shutdown.')
        if platform.system().lower() == 'linux':
            os.system("shutdown -h now");
        else:#windows
            os.system("shutdown /s /t 1");