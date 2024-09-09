import torch


def get_loss_fn(name):
    if name == 'binary_cross_entropy':
        result = torch.nn.BCELoss()
    else:
        result = torch.nn.BCELoss()
    return result


def get_optimizer(name, model_params, learn_rate, weight_decay=None, momentum=None):
    if name == 'adam':
        if weight_decay is not None:
            result = torch.optim.Adam(params=model_params, lr=learn_rate, weight_decay=weight_decay)
        else:
            result = torch.optim.Adam(params=model_params, lr=learn_rate)
    elif name == 'sgd' and momentum is not None:
        if momentum is not None:
            result = torch.optim.SGD(params=model_params, lr=learn_rate, momentum=momentum)
        else:
            result = torch.optim.SGD(params=model_params, lr=learn_rate)
    else:
        result = torch.optim.Adam(params=model_params)
    return result


def get_scheduler(name, optimizer):
    if name == 'reduce_on_plateau':
        result = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    else:
        result = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    return result
