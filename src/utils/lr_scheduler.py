from timm.scheduler import CosineLRScheduler

def build_scheduler(config, optimizer, n_iter_per_epoch, total_iters=None, warmup_iters=None):
    """
    Iteration-based LR scheduler.

    Args:
        config: config.train (expected to have lr_scheduler, min_lr, warmup_lr, etc.)
        optimizer: optimizer
        n_iter_per_epoch: used to compute defaults if total_iters/warmup_iters not provided
        total_iters: total training iterations (preferred)
        warmup_iters: warmup iterations (preferred)
    """
   
    lr_scheduler = None

    if config.lr_scheduler.name == 'cosine':

        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial= config.total_cosine_decay_iter if hasattr(config, 'total_cosine_decay_iter') else 100000, #int(total_iters),
            lr_min=config.min_lr,
            warmup_lr_init= config.warmup_lr if hasattr(config, 'warmup_lr') else 0.0,
            warmup_t= config.warmup_iter if hasattr(config, 'warmup_iter') else 0,
            t_in_epochs=False, 
        )

    else:
        raise NotImplementedError(f'lr scheduler {config.lr_scheduler.name} not implemented')

    print(f'Built lr_scheduler: {lr_scheduler}')
    return lr_scheduler
