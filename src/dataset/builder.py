import torch
import pandas as pd
import random
import numpy as np
from src.utils import get_logger
from .dataset import DeadwoodDataset
from .capped_sampler import CappedSampler
from .random_sampler import FixedSizeRandomSampler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
    
def collate_fn(batch):
    """Collate: drop failed samples and stack tensors."""
    batch = [b for b in batch if b is not None]
    
    if len(batch) == 0:
        return None
    
    assert 'image' in batch[0], "Expected 'image' key in dataset output"
    assert 'mask' in batch[0], "Expected 'mask' key in dataset output"

    images = torch.stack([b['image'] for b in batch])
    masks  = torch.stack([b['mask']  for b in batch])
    

    return {
        'image': images,
        'mask': masks,
        'ortho_id': [b["ortho_id"] for b in batch],
        'path': [b["path"] for b in batch],
        'idx': [b["idx"] for b in batch]
    }


def _parse_dataset(meta):
    '''Parse train/val CSVs into dataframes.
       Expects meta.train and meta.test to be paths to CSV files with columns 'ortho_id',
      'image_path', 'mask_path', and optionally stratification columns for sampling.
    '''
    train_df = pd.read_csv(meta.train)
    val_df   = pd.read_csv(meta.test)
    return train_df, val_df


def _build_dataset(config):
    train_df, val_df = _parse_dataset(config.meta)
    
    dataset_train = DeadwoodDataset(dataframe=train_df, input_dir=config.input_dir, crop_size=config.crop_size, is_train=True)
    dataset_val = DeadwoodDataset(dataframe=val_df, input_dir=config.input_dir, crop_size=None, is_train=False)
    
    return dataset_train, dataset_val


def _build_sampler(config, dataset_train, dataset_val):
    
    if getattr(config, "weighted_col", None) is not None:
        sampler_train = CappedSampler(
            dataframe=dataset_train.df,
            stratify_col=config.weighted_col,       
            shuffle=True,
            seed=config.data_sampling_seed,
            log=True,
            max_samples_per_group= getattr(config, "max_samples_per_group", None)
        )
    else:
        sampler_train = FixedSizeRandomSampler(
        dataset=dataset_train,
        num_samples= len(dataset_train),     #28428,  # match capped sampler
        seed=config.data_sampling_seed,
        shuffle=True,
        log=True
        )
        # sampler_train = RandomSampler(dataset_train, replacement=False)

    sampler_val = SequentialSampler(dataset_val)
    return sampler_train, sampler_val

def _build_dataloader(config, dataset_train, dataset_val, sampler_train, sampler_val):
    num_workers = getattr(config.data, "num_workers", 8)
    
    generator = torch.Generator()
    generator.manual_seed(config.seed)

    data_loader_train = DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=config.data.batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers= num_workers > 0,  # Only use persistent workers if num_workers > 0
        prefetch_factor= 2 if num_workers > 0 else None,  # Avoid prefetching if num_workers=0
        collate_fn=collate_fn,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        generator=generator
    )

    data_loader_val = DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=getattr(config.data, "inference_batch_size", 1),
        num_workers=getattr(config.data, "num_workers_val", 4),
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn,
        drop_last=False
    )
    return data_loader_train, data_loader_val

def build_loader(config):
    """
    Build train/val datasets and dataloaders
    """
    logger = get_logger()
    logger.info("Building datasets and dataloaders...")
    # -------------------------------
    # Parse datasets
    # -------------------------------
    dataset_train, dataset_val = _build_dataset(config.data)
    logger.info(f"Successfully built train dataset: {len(dataset_train)} samples")
    logger.info(f"Successfully built val dataset: {len(dataset_val)} samples")

    # -------------------------------
    # Samplers
    # -------------------------------
    sampler_train, sampler_val = _build_sampler(config.data, dataset_train, dataset_val)
    logger.info(f"Train sampler type: {type(sampler_train).__name__}")
    logger.info(f"Val sampler type: {type(sampler_val).__name__}")
    logger.info(f"Successfully built train sampler: {len(sampler_train)} samples")
    logger.info(f"Successfully built val sampler: {len(sampler_val)} samples")

    # -------------------------------
    # Dataloaders
    # -------------------------------
    data_loader_train, data_loader_val = _build_dataloader(config, dataset_train, dataset_val, sampler_train, sampler_val)
    logger.info(f"Successfully built train dataloader: {len(data_loader_train)} batches")
    logger.info(f"Successfully built val dataloader: {len(data_loader_val)} batches")
    
    return dataset_train, data_loader_train, data_loader_val, sampler_train

def build_loader_inference(config):
    """
    Build dataloader for inference on test set. Uses SequentialSampler and batch size specified by config.data.inference_batch_size.
    """
    
    logger = get_logger()
    logger.info("Building dataset and dataloader for inference...")
    
    val_df   = pd.read_csv(config.data.meta.test)
    dataset_val = DeadwoodDataset(dataframe=val_df, input_dir=config.data.input_dir, crop_size=None, is_train=False)
    logger.info(f"Successfully built val dataset: {len(dataset_val)} samples")

    # Sampler
    sampler_val = SequentialSampler(dataset_val)
    logger.info(f"Val sampler type: {type(sampler_val).__name__}")
    logger.info(f"Successfully built val sampler: {len(sampler_val)} samples")

    # Dataloader
    data_loader_val = DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=getattr(config.data, "inference_batch_size", 1),
        num_workers=getattr(config.data, "num_workers_val", 4),
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn,
        drop_last=False
    )
    logger.info(f"Successfully built val dataloader: {len(data_loader_val)} batches")
    
    return data_loader_val

