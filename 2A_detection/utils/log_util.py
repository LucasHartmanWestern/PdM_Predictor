import logging


def setup_basic_logger(file_path):
    logging.basicConfig(
        filename=file_path,
        filemode='w',
        format="%(message)s",
        level=logging.INFO,
    )


def log(msg):
    logging.info(msg)


def log_and_print(msg):
    logging.info(msg)
    print(msg)


def print_hyperparams(**kwargs):
    log_and_print('Hyperparameters:')
    for key, value in kwargs.items():
        log_and_print('\t{}: {}'.format(key, str(value)))
