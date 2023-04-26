from torchvision import transforms
from transforms.co_transforms import get_co_transforms
from transforms.ar_transforms.ap_transforms import get_ap_transforms
from transforms import sep_transforms
from Dataset.flow_datasets import UnsupervisedOpticalDataset, UnsupervisedOpticalDatasetValid
import torch


def get_dataset(all_cfg):
    cfg = all_cfg.data

    input_transform = transforms.Compose([
        sep_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
    ])

    co_transform = get_co_transforms(aug_args=all_cfg.data_aug)

    ap_transform = get_ap_transforms(cfg.at_cfg) if cfg.run_at else None

    train_set = UnsupervisedOpticalDataset(cfg.root_dataset,
                                           ap_transform=ap_transform,
                                           transform=input_transform,
                                           co_transform=co_transform
                                           )
    test_set = UnsupervisedOpticalDataset(cfg.root_dataset_test,
                                          ap_transform=None,
                                          transform=input_transform,
                                          co_transform=None
                                          )
    valid_set = UnsupervisedOpticalDatasetValid(cfg.root_dataset_val,
                                                ap_transform=None,
                                                transform=input_transform,
                                                co_transform=None
                                                )
    return train_set, valid_set, test_set


def get_dataloader(cfg):
    train_set, valid_set, test_set = get_dataset(cfg)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=cfg.train.batch_size,
        num_workers=cfg.train.workers, pin_memory=True, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=min(1, cfg.train.batch_size),
        num_workers=min(4, cfg.train.workers),
        pin_memory=True, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=min(1, cfg.train.batch_size),
        num_workers=min(4, cfg.train.workers),
        pin_memory=True, shuffle=False)
    return train_loader, test_loader
