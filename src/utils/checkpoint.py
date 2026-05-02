import os
import torch
from src.utils import get_logger
from safetensors.torch import save_file

    

def save_checkpoint(config, epoch, model, suffix=''):
    '''
    Save model weights only in safetensors format. Filename includes epoch and optional suffix (e.g. best_macro_f1).
    '''
    logger = get_logger()

    if len(suffix) > 0 and not suffix.startswith('_'):
        suffix = '_' + suffix
        safetensor_name = f'ckpt_epoch_{epoch}{suffix}.safetensors'
    else:
        safetensor_name = f'model_{epoch}.safetensors'

    safetensor_path = os.path.join(config.output, safetensor_name)

    os.makedirs(config.output, exist_ok=True)

    logger.info(f'Saving model weights only: {safetensor_path}')
    model_state = model.state_dict()
    model_state = {k: v.contiguous() if isinstance(v, torch.Tensor) and not v.is_contiguous() else v
            for k, v in model_state.items()}
    save_file(model_state, safetensor_path)

