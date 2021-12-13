import logging.config
import config
import logging


class Logger:
    def __init__(self):
        self.log_conf = config.log_conf

    def get_logger(self, name='root'):
        logging.config.fileConfig(self.log_conf)
        return logging.getLogger(name)


def get_assistant_logger(name='root'):
    logger = Logger()
    logger = logger.get_logger(name)
    return logger


def make_log():
    logger = logging.getLogger('assistant')
    logger.setLevel(logging.DEBUG)  # 笔的日志级别高
    """
        处理器(纸)
    """
    consoleHander = logging.StreamHandler()  # 打印控制台
    consoleHander.setLevel(logging.DEBUG)

    # 不知道打印级别默认使用logger的
    fileHander = logging.FileHandler(filename=config.log_file, mode='a')
    fileHander.setLevel(logging.DEBUG)

    # formatter格式
    formatter = logging.Formatter(fmt='%(asctime)s|%(levelname)-8s|%(filename)10s|%(funcName)10s|%(lineno)4s|%(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    # 给处理器设置格式
    consoleHander.setFormatter(formatter)
    fileHander.setFormatter(formatter)

    # 记录器设置写到哪些处理器
    logger.addHandler(consoleHander)
    logger.addHandler(fileHander)
    return logger


logger = make_log()


if __name__ == '__main__':
    logger_conf = get_assistant_logger('assistant')
    logger_conf.info('hello world')
    logger.info('hello world')
