"""
Training script for segmentation backbone
"""
import argparse
import os
import time
import math
import datetime
import warnings
import torch
import sys
sys.path.append('./src')  # Ensure src is in path for imports
from timm.utils import AverageMeter
from omegaconf import OmegaConf
from src.loss import TverskyFocalLoss
from src.dataset import build_loader
from src.model import build_model
from src.utils import (
    get_config,
    get_logger,
    build_optimizer,
    build_scheduler,
    set_random_seed,
    GlobalSegmentationMetrics,
    OrthoSegmentationMetrics,
    save_checkpoint
)


warnings.filterwarnings("ignore", category=UserWarning)



def parse_args():
    '''
    Parse command line arguments for training and evaluation.
     - --cfg: path to config file (required)
     - --opts: optional list of KEY=VALUE pairs to override config options
     - --batch-size: override batch size for single GPU
     - --seed: random seed for reproducibility
     - --output: root output directory for checkpoints and logs
     - --tag: experiment tag for logging (e.g. with W&B)
     - --wandb: flag to enable W&B logging
    '''
    
    parser = argparse.ArgumentParser('PANOPS training and evaluation script')
    parser.add_argument('--cfg', type=str, required=True, help='path to config file')
    parser.add_argument('--opts', help="Modify config options by adding 'KEY=VALUE' list. ", default=None, nargs='+')
    parser.add_argument('--batch-size', type=int, help='batch size for single GPU')
    parser.add_argument('--seed', type=int, default=0, help='random seed for initialization')
    parser.add_argument('--output',default='/home/as2114/code/benchmark_deadtree/output/resnet34',type=str, help='root of output folder')
    parser.add_argument('--tag', type=str, help='tag of experiment')
    parser.add_argument('--wandb', action='store_true', help='Use W&B to log experiments')
    return parser.parse_args()


def train_one_epoch(cfg, model, data_loader, optimizer, epoch, lr_scheduler, criterion):
    '''
    Train the model for one epoch.
     - Uses automatic mixed precision for efficiency
     - Tracks time and loss with AverageMeter
     - Logs progress every cfg.print_freq iterations
     - Logs to W&B if enabled in config
    '''
    
    # Set model to training mode
    model.train()
    
    # PyTorch's automatic mixed precision 
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())
    
    # Meters for tracking time and loss
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    
    # Logger
    logger = get_logger()
    
    # Tracking time 
    start = time.time()
    end = start
    
    # Get the device
    device = next(model.parameters()).device
    
    # Number of iterations in this epoch 
    num_steps = len(data_loader)

    # W&B setup
    if hasattr(cfg, 'wandb') and cfg.wandb:
        import wandb
    else:
        wandb = None

    # Iterate over batches
    for idx, samples in enumerate(data_loader):
        
        # Zero the gradients
        optimizer.zero_grad()

        # Move data to device
        images = samples['image'].to(device=device, non_blocking=True)
        masks_true = samples['mask'].to(device=device, dtype=torch.long, non_blocking=True).squeeze(1)

        # Forward pass with automatic mixed precision
        with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
            masks_pred = model(images)
            loss = criterion(masks_pred, masks_true)
        
        # Backward pass and optimization step with gradient scaling
        scaler.scale(loss).backward()
        
        # Gradient clipping if specified in config after unscaling
        if cfg.train.get('clip_grad', None):
            scaler.unscale_(optimizer)  # IMPORTANT before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip_grad)

       # optimizer.step()
        scaler.step(optimizer)
        scaler.update()

        # Update learning rate scheduler (iteration-based)
        lr_scheduler.step_update(epoch * num_steps + idx)

        # Update loss meter
        loss_meter.update(loss.item())

        # Update batch time meter
        batch_time.update(time.time() - end)
        end = time.time()

        # Log progress every cfg.print_freq iterations
        if idx % cfg.print_freq == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0) if torch.cuda.is_available() else 0
            etas = batch_time.avg * (num_steps - idx)
            logger.info(f'Train: [{epoch}/{cfg.train.epochs}][{idx}/{num_steps}]'
                        f' eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}'
                        f' time {batch_time.val:.4f} ({batch_time.avg:.4f})'
                        f' total_loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})'
                        f' mem {memory_used:.0f}MB')

            if wandb is not None:
                log_stat = {}
                log_stat['iter/loss'] = loss_meter.avg
                log_stat['iter/learning_rate'] = lr
                wandb.log(log_stat)

    epoch_time = time.time() - start
    logger.info(f'EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}')
    return dict(total_loss=loss_meter.avg)


@torch.inference_mode()
def validate(config, data_loader, model, criterion):
    # Logger
    logger = get_logger()
    model.eval()
    use_wandb = config.wandb if hasattr(config, 'wandb') else False
    if use_wandb:
        import wandb
    else:
        wandb = None

    device = next(model.parameters()).device
    batch_time = AverageMeter()
    loss_meter = AverageMeter()

    end = time.time()
    logger.info("Inference begins")
    logger.info(f"len of data_loader:{len(data_loader)}")

    # 🔸 Initialize global metrics accumulator
    metrics = GlobalSegmentationMetrics(num_classes=config.model.num_classes , ignore_index=255, device=device)
    ortho_metrics = OrthoSegmentationMetrics(num_classes=config.model.num_classes , ignore_index=255, device =device)
    
    for idx, samples in enumerate(data_loader):

        images = samples["image"].to(device=device, dtype=torch.float32)
        masks_true = samples["mask"].to(device=device, dtype=torch.long).squeeze(1)
        ortho_ids = samples["ortho_id"]

        masks_pred = model(images)

        # loss
        loss = criterion(masks_pred, masks_true)
        loss_meter.update(loss.item(), 1)

        # metrics update
        if criterion.logits:
            preds = torch.softmax(masks_pred, dim=1).argmax(dim=1)
        else:
            preds = masks_pred.argmax(dim=1)
            
        metrics.update(preds, masks_true)
        
        for b in range(preds.shape[0]):
            ortho_metrics.update(
                preds[b],
                masks_true[b],
                ortho_id=ortho_ids[b]
            )

        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.print_freq == 0:
            memory_used = (
                torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                if torch.cuda.is_available()
                else 0
            )
            logger.info(
                f"ValIter: {idx} | time {batch_time.val:.4f} ({batch_time.avg:.4f}) "
                f"loss {loss_meter.avg:.4f} mem {memory_used:.0f}MB"
            )

    results = metrics.compute()
    ortho_results = ortho_metrics.compute()

    mean_precision = float(results["precision"].mean())
    mean_recall = float(results["recall"].mean())
    mean_f1 = results["mean_f1"]
    mean_iou = results["mean_iou"]
    
    logger.info(f"mean precision: {mean_precision}, mean recall: {mean_recall}, mean f1: {mean_f1}, mean iou: {mean_iou}")
    
    macro_f1 = ortho_results["macro_f1"]
    
    # For DeadTree class (c=2), log separately, rare and important class for our applicatiom.
    DEADTREE_CLASS_IDX = 2

    mean_f1_c2 = results["f1"][DEADTREE_CLASS_IDX].item()
    mean_recall_c2 = results["recall"][DEADTREE_CLASS_IDX].item()

    macro_f1_c2 = ortho_results["macro_f1_per_class"][DEADTREE_CLASS_IDX].item()
    macro_recall_c2 = ortho_results["macro_recall_per_class"][DEADTREE_CLASS_IDX].item()
    
    logger.info(
    f"[DeadTree c=2] mean_f1={mean_f1_c2:.4f}, mean_recall={mean_recall_c2:.4f}, "
    f"macro_f1={macro_f1_c2:.4f}, macro_recall={macro_recall_c2:.4f}"
    )


    # 🔸 WandB logging
    if wandb is not None:
        log_dict = {
            "val/loss": loss_meter.avg,
            "val/mean_f1": mean_f1,
            "val/mean_precision": mean_precision,
            "val/mean_recall": mean_recall,
            "val/mean_iou": mean_iou,
            "val/macro_f1": macro_f1
            
        }
        for cls_idx in range(len(results["precision"])):
            log_dict[f"val/class_{cls_idx}_precision"] = results["precision"][cls_idx].item()
            log_dict[f"val/class_{cls_idx}_recall"] = results["recall"][cls_idx].item()
            log_dict[f"val/class_{cls_idx}_f1"] = results["f1"][cls_idx].item()
            log_dict[f"val/class_{cls_idx}_iou"] = results["iou"][cls_idx].item()
            log_dict[f"val/class_{cls_idx}_macro_f1"] = ortho_results['macro_f1_per_class'][cls_idx].item()
            log_dict[f"val/class_{cls_idx}_macro_recall"] = ortho_results['macro_recall_per_class'][cls_idx].item()
            log_dict[f"val/class_{cls_idx}_macro_precision"] = ortho_results['macro_precision_per_class'][cls_idx].item()
            log_dict[f"val/class_{cls_idx}_macro_iou"] = ortho_results['macro_iou_per_class'][cls_idx].item()
        wandb.log(log_dict)

    logger.info(
        f"Validation Summary: mean F1 {mean_f1:.4f}, mean IoU {mean_iou:.4f}, loss {loss_meter.avg:.4f}"
    )

    return (
        mean_f1,
        mean_precision,
        mean_recall,
        mean_iou,
        loss_meter.avg,
        macro_f1,
        macro_f1_c2,

    )


def get_criterion(config):
    '''
    Factory function to get loss criterion based on config.
     - Supports TverskyFocalLoss and DiceLoss (as special case of Tversky with alpha=beta=0.5, gamma=1.5)
    '''
    # config: expects config.type and possibly other params
    t = config.type if hasattr(config, "type") else None
    if t == "tverskyfocal":
        # assume TverskyFocalLoss supports num_classes argument for multiclass
        return TverskyFocalLoss(num_classes=config.num_classes, alpha=config.alpha, beta=config.beta, gamma=config.gamma, logits=config.logits)
    elif t == "dice":
        return TverskyFocalLoss(num_classes=config.num_classes, alpha=0.5, beta=0.5, gamma=1.5, logits=config.logits)
    else:
        raise ValueError(f"Unknown loss type: {t}")



def train(cfg):
    
    logger = get_logger()

    if cfg.wandb:
        import wandb
        wandb.init(
            project='deadtree_seg_v3',
            name=os.path.join(cfg.model_name, cfg.tag) if hasattr(cfg, 'model_name') else cfg.tag,
            dir=cfg.output if hasattr(cfg, 'output') else '.',
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # seed & cudnn
    set_random_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  

    # build data, model, optimizer, scheduler
    _, data_loader_train, data_loader_val, sampler_train = build_loader(cfg)
    model = build_model(cfg.model).to(device=device)
    optimizer = build_optimizer(cfg.train, model)
    
    n_iter_per_epoch =  len(data_loader_train)
    lr_scheduler = build_scheduler(cfg.train, optimizer, n_iter_per_epoch=n_iter_per_epoch)
    criterion = get_criterion(cfg.train.loss).to(device=device)

    print("===================================================================")
    print(f"train_dataloader: {data_loader_train}")
    print(f"train_dataloader.sampler: {sampler_train}")
    print(f"train_dataloader.batch_sampler: {data_loader_train.batch_sampler}")
    print("===================================================================")

    # Metrics tracking
    max_metrics = {
        'max_val_f1': -math.inf,
        'min_val_loss': math.inf,
        'max_precision': -math.inf,
        'max_recall': -math.inf,
        "max_macro_f1": -math.inf,
        "max_macro_recall": -math.inf,
        "max_mean_f1_c2": -math.inf,
        "max_mean_recall_c2": -math.inf,
        "max_macro_f1_c2": -math.inf,
        "max_macro_recall_c2": -math.inf,
    }

    # --- Early stopping setup ---
    patience_counter = 0
    patience_limit = cfg.patience_limit   # number of eval rounds to wait

    # -----------------------------

    start_time = time.time()
    logger.info(f'Start training from epoch {cfg.train.start_epoch} to {cfg.train.epochs - 1}')
    EPS = cfg.patience_threshold  # minimum improvement to reset patience
    for epoch in range(cfg.train.start_epoch, cfg.train.epochs):
        logger.info(f'Setting epoch and updating indices in sampler')
        sampler_train.set_epoch(epoch) if hasattr(sampler_train, 'set_epoch') else None
        torch.cuda.empty_cache()
        loss_train_dict = train_one_epoch(cfg, model, data_loader_train, optimizer, epoch, lr_scheduler, criterion)
        loss_train = loss_train_dict['total_loss']
        logger.info(f'Avg loss of the network on the train set: {loss_train:.4f}')

        # Save intermediate checkpoints
        if epoch % cfg.checkpoint.save_freq == 0:
            save_checkpoint(
                config=cfg,
                epoch=epoch,
                model=model,
                suffix=''
            )

        # Validation
        if (epoch % cfg.evaluate.eval_freq == 0) or (epoch == cfg.train.epochs - 1) or (epoch == 0):
            (
                f1, precision, recall, iou, val_loss,
                macro_f1, 
                macro_f1_c2
            ) = validate(
                            cfg, data_loader_val, model, criterion
                        )
            

            logger.info(
                f"VAL meanF1: {f1:.4f} IoU: {iou:.4f} "
                f"precision: {precision:.4f}, recall={recall:.4f} "
                f"val_loss: {val_loss:.4f}"
            )
            saved = False
            # Improvement criteria:
            # - Primary: macro F1 (global robustness)
            # - Secondary: class-2 (dead tree) macro F1 (task-specific importance)
            # to avoid saving for negligible improvements
            improved_macro = macro_f1 > max_metrics["max_macro_f1"] + EPS
            improved_c2 = macro_f1_c2 > max_metrics["max_macro_f1_c2"] + EPS

            saved = False

            if improved_macro:
                suffix = "best_macro_f1"
                saved = True
            elif improved_c2:
                suffix = "best_deadtree_macro_f1"
                saved = True
                
            if improved_macro:
                max_metrics["max_macro_f1"] = macro_f1

            if improved_c2:
                max_metrics["max_macro_f1_c2"] = macro_f1_c2

                
            # max_metrics["max_macro_recallax(max_metrics["max_macro_f1"], macro_f1)
            # max_metrics["max_macro_recall"] = max(max_metrics["max_macro_recall"], macro_recall)
            if saved:
                patience_counter = 0
                save_checkpoint(
                    config=cfg,
                    epoch=epoch,
                    model=model,
                    suffix=suffix,
                )
            else: 
                patience_counter +=1
                logger.info(f"No improvement. Patience {patience_counter}/{patience_limit}")
                if patience_counter >= patience_limit:
                    logger.info("Early stopping triggered.")
                    break


            if val_loss < max_metrics['min_val_loss']:
                max_metrics['min_val_loss'] = val_loss

    total_time = time.time() - start_time
    logger.info('Training time {}'.format(str(datetime.timedelta(seconds=int(total_time)))))

if __name__ == '__main__':
    args = parse_args()
    config = get_config(args)
    logger = get_logger(config)

    os.makedirs(config.output, exist_ok=True)

    train(config)
