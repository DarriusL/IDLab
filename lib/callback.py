# @Time   : 2023.03.03
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
import logging, os, inspect, smtplib
from email.mime.text import MIMEText
from email.header import Header

class CustomException(Exception):
    ''' Used to output exception information if an exception is thrown

    Parameters:
    -----------

    Exception: str
        The exception information output when an exception is actively thrown
    
    Examples:
    ---------
    >>> import lib.callback.CustomException as ce
    >>> raise ce('error')

    '''
    def __init__(self,ErrorInfo):
        self.ErrorInfo = '\033[0;31m' + str(ErrorInfo) + '\033[0m';
    def __str__(self):
        return self.ErrorInfo


class Logger():
    '''Logger class: logging that can be output on the console and logger files at the same time

    Parameters:
    -----------

    level:
    Default level of logging output
    default: logging.INFO

    filename:
    path of the logger file
    default:'./cache/logger/logger.log'

    Example:
    --------
    >>> import lib.callback.Logger
    >>> log = Logger().get_log()
    >>> log.info('Example complete.')
    '''
    def __init__(self, level = logging.INFO, filename = './cache/logger/logger.log') -> None:
        #format for the logger
        self.format = logging.Formatter(fmt = '\n[%(asctime)s]-[%(levelname)s - %(name)s] \n%(message)s');
        self.level = level;
        path, _ = os.path.split(filename);
        if not os.path.exists(path):
            raise CustomException(f'Path({path}) does not exist');
        self.filename = filename;
        #Create a logger object
        self.logger = logging.getLogger(__name__);
        self.logger.setLevel(self.level);
        self.logger.addHandler(self.get_console_handler());
        self.logger.addHandler(self.get_file_handler());

    def get_console_handler(self):
        #Create a log handler for the console
        console_handler = logging.StreamHandler();
        console_handler.setLevel(self.level);
        console_handler.setFormatter(self.format);
        return console_handler;

    def get_file_handler(self):
        #Create a log handler for the file
        file_handler = logging.FileHandler(
            filename = self.filename,
            mode = 'w'
        )
        file_handler.setLevel(self.level);
        file_handler.setFormatter(self.format);
        return file_handler;

    def get_log(self):
        return self.logger;


def send_smtp_emil(sender, receiver, passward, subject, content, port = 25, server = None):
    '''Send email using SMTP protocol

    Parameter:
    ----------
    sender:str
        Address to send email
    
    receiver:str
        Address to receive email from sender

    passward:str
        sender's password or authorization code
    
    subject:str
        email subject

    content:str
        email content

    port:int, optional
        port of server
        default:25
    
    server:str, optional
        default:None
    '''
    smtp = smtplib.SMTP();
    if server is None:
        server = 'smtp.'+sender.split('.')[-2].split('@')[-1]+'.com';
    smtp.connect(server, port);
    smtp.login(sender, passward);
    message = MIMEText(content, 'plain', 'utf-8');
    message['Subject'] = Header(subject, 'utf-8');
    smtp.sendmail(sender, receiver, message.as_string());
    smtp.quit()

class VarReporter():
    '''Variable reporter, mainly used to output parameters during training
    
    Example:
    --------
    >>> from lib.callback import VarReporter
    >>> reporter = VarReporter()
    >>> var_1 = 1; var_2 = 2;
    >>> reporter.add('parameter of fun 1', var_1)
    >>> reporter.add('parameter of fun 1', var_1)
    >>> ...
    >>> reporter.report(logger, head);
    '''
    def __init__(self) -> None:
        self.var_dict = {};

    def add(self, key, value):
        '''Add variables that need to be reported to reporter
        '''
        self.var_dict[key] = value;

    def report(self, logger_func, head, num_of_var_per_line = 4):
        '''Reporting parameters

        Parameters:
        -----------
        logger_func:function
            logger function
            e.g. logger.info
        
        head:str
            content preamble

        var_dict:dict
            A dictionary of parameters to print
        
        num_of_var_per_line:int, optional
            number of arguments to print per line
            default:4
        '''
        content = head + '\n';
        cnt = 0;
        for key, value in self.var_dict.items():
            content += f'{key}:[{value}] ';
            cnt += 1;
            if cnt >= num_of_var_per_line:
                cnt = 0;
                content += '\n';
        logger_func(content);
