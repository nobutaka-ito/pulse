import torch

def make_CNN(blocks, channels, droprate, fcblocks, method):

    modules = []
    modules.append(torch.nn.Conv2d(1, channels, 3, padding = 'same'))
    modules.append(torch.nn.ReLU())
    if method == 'PN':
        modules.append(torch.nn.BatchNorm2d(channels))
    else:
        modules.append(torch.nn.Dropout2d(droprate))
    modules.append(torch.nn.Conv2d(channels, channels, 3, padding = 'same'))
    modules.append(torch.nn.ReLU())
    if method == 'PN':
        modules.append(torch.nn.BatchNorm2d(channels))
    else:
        modules.append(torch.nn.Dropout2d(droprate))
    for i in range(blocks - fcblocks):
        modules.append(torch.nn.Conv2d(channels * (2 ** i), channels * (2 ** (i + 1)), 3, padding = 'same'))
        modules.append(torch.nn.ReLU())
        if method == 'PN':
            modules.append(torch.nn.BatchNorm2d(channels * (2 ** (i + 1))))
        else:
            modules.append(torch.nn.Dropout2d(droprate))
        modules.append(torch.nn.Conv2d(channels * (2 ** (i + 1)), channels * (2 ** (i + 1)), 3, padding = 'same'))
        modules.append(torch.nn.ReLU())
        if method == 'PN':
            modules.append(torch.nn.BatchNorm2d(channels * (2 ** (i + 1))))
        else:
            modules.append(torch.nn.Dropout2d(droprate))
    for i in range(blocks - fcblocks, blocks):
        modules.append(torch.nn.Conv2d(channels * (2 ** i), channels * (2 ** (i + 1)), 1, padding = 'same'))
        modules.append(torch.nn.ReLU())
        if method == 'PN':
            modules.append(torch.nn.BatchNorm2d(channels * (2 ** (i + 1))))
        else:
            modules.append(torch.nn.Dropout2d(droprate))
        modules.append(torch.nn.Conv2d(channels * (2 ** (i + 1)), channels * (2 ** (i + 1)), 1, padding = 'same'))
        modules.append(torch.nn.ReLU())
        if method == 'PN':
            modules.append(torch.nn.BatchNorm2d(channels * (2 ** (i + 1))))
        else:
            modules.append(torch.nn.Dropout2d(droprate))

    if method == 'MixIT':
        modules.append(torch.nn.Conv2d(channels * (2 ** blocks), 3, 3, padding = 'same'))
    elif method == 'PN':
        modules.append(torch.nn.Conv2d(channels * (2 ** blocks), 1, 3, padding = 'same'))
    else:
        modules.append(torch.nn.Conv2d(channels * (2 ** blocks), 1, 1, padding = 'same'))
    
    return torch.nn.Sequential(*modules)