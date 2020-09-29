from .RefineDet import RefineDet512, RefineDetLoss
from .RefineDetTraffic import RefineDetTraffic, RefineDetTrafficLoss


def model_entry(config):
    if config.model['arch'].upper() == 'REFINEDET':
        print('Loading RefineDet512 model with VGG-16 backbone ......')
        return RefineDet512(config['n_classes'], config=config), RefineDetLoss
    elif config.model['arch'].upper() == 'REFINEDETTRAFFIC':
        print('Loading RefineDet with VGG-16 backbone, DETRAC finetuned model ......')
        return RefineDetTraffic(config['n_classes'], config=config), RefineDetTrafficLoss
    else:
        print('Only RefineDet is supported.')
        raise NotImplementedError
