from torch import optim
from src.utils import get_logger


def build_optimizer(config, model):
    """Build optimizer"""
    logger = get_logger()

    parameters = set_weight_decay(model)

    if config.optimizer.name == "adamw":
        optimizer = optim.AdamW(
            parameters,
            lr=config.optimizer.base_lr,
            eps=config.optimizer.eps,
            betas=config.optimizer.betas,
            weight_decay=config.weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer.name}")

    logger.info(f"Built optimizer: {optimizer}")
    return optimizer


def set_weight_decay(model, skip_list=(), skip_keywords=()):
    logger = get_logger()

    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if (
            len(param.shape) == 1
            or name.endswith(".bias")
            or name in skip_list
            or any(k in name for k in skip_keywords)
        ):
            no_decay.append(param)
        else:
            has_decay.append(param)

    if len(has_decay) + len(no_decay) == 0:
        raise ValueError("No trainable parameters found")

    logger.info(
        f"Params with decay: {sum(p.numel() for p in has_decay)}, "
        f"without decay: {sum(p.numel() for p in no_decay)}"
    )

    return [
        {"params": has_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def check_keywords_in_name(name, keywords=()):
    return any(keyword in name for keyword in keywords)