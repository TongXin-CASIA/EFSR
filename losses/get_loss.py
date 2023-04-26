from .flow_loss import unFlowLoss, MaskedFlowLoss


def get_loss(cfg):
    if cfg.type == 'unflow':
        loss = unFlowLoss(cfg)
    elif cfg.type == 'masked_flow':
        loss = MaskedFlowLoss(cfg)
    else:
        raise NotImplementedError(cfg.type)
    return loss
