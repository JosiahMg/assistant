import logging.config
import config


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


if __name__ == '__main__':
    logger = get_assistant_logger('assistant')
    logger.info('hello world')
