import torch
import torch.distributed as dist
import torchvision.transforms as T
from collections import defaultdict

from torch.utils.data import DataLoader
from .bases import ImageDataset
from .preprocessing import RandomErasing
from .sampler import RandomIdentitySampler
from .aicV2 import AIC, AIC_UDA
from .aic_sim import AIC_SIM
from .aic_sim_spgan import AIC_SIM_SPGAN
from .sim_view import SIM_VIEW
from .sampler_ddp import RandomIdentitySampler_DDP
from .autoaugment import AutoAugment


__factory = {
    'aic': AIC,
    #'aic': AIC_UDA,
    'aic_sim': AIC_SIM,
    'aic_sim_spgan': AIC_SIM_SPGAN,
    'sim_view': SIM_VIEW,
}

def train_collate_fn(batch):
 
  
    imgs, pids, camids, _ , _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids

def val_collate_fn(batch):   ##### revised by luo
    imgs, pids, camids, trackids, img_paths = zip(*batch)
    #  camidssss = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, trackids, img_paths

def zero():
    return 0
    
def class_balance_weight(list_pids):
    class_weight_dict = defaultdict(zero)
    for pid in list_pids:
        class_weight_dict[pid]+=1
    class_weights = []
    for key in sorted(class_weight_dict.keys()):
         class_weights.append(class_weight_dict[key])
    return class_weights

def make_dataloader(cfg):
    if cfg.INPUT.RESIZECROP == True:
        randomcrop = T.RandomResizedCrop(cfg.INPUT.SIZE_TRAIN,scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3)
    else:
        randomcrop = T.RandomCrop(cfg.INPUT.SIZE_TRAIN)
    train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING, padding_mode='constant'), # input size = [394,394]
            randomcrop, # input size = [384,384]
            T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.15, saturation=0, hue=0)], p=cfg.INPUT.COLORJIT_PROB),
            T.ToTensor(),
            T.RandomErasing(p=cfg.INPUT.RE_PROB, value=cfg.INPUT.PIXEL_MEAN),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        ])
    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST, interpolation=3),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = __factory[cfg.DATASETS.NAMES](root = cfg.DATASETS.ROOT_DIR,
                                            crop_test = cfg.TEST.CROP_TEST)
                                            

    class_weights = class_balance_weight(dataset.list_train_pids+dataset.list_vali_pids)

    train_set = ImageDataset(dataset.train+dataset.vali, train_transforms)
    num_classes = dataset.num_train_pids+dataset.num_vali_pids
    if 'triplet' in cfg.DATALOADER.SAMPLER:

        if cfg.MODEL.DIST_TRAIN:
            print('DIST_TRAIN START')
            mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // dist.get_world_size()
            data_sampler = RandomIdentitySampler_DDP(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
            batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
            train_loader = torch.utils.data.DataLoader(
                train_set,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=train_collate_fn,
                pin_memory=True,
            )
        else:
            train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=RandomIdentitySampler(dataset.train, 
                                              cfg.SOLVER.IMS_PER_BATCH, 
                                              cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=train_collate_fn
            )
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))

    if cfg.DATASETS.QUERY_MINING:
        val_set = ImageDataset(dataset.query + dataset.query, val_transforms)
    else:
        #val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
        val_set = ImageDataset(dataset.vali, val_transforms)
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False,
        num_workers=num_workers, collate_fn=val_collate_fn
    )

    return train_loader, val_loader, 0, num_classes, class_weights


def make_dataloader_Pseudo(cfg):
    train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])
    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])
    print('using size :{} for training'.format(cfg.INPUT.SIZE_TRAIN))

    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR,plus_num_id=cfg.DATASETS.PLUS_NUM_ID)
    num_classes = dataset.num_train_pids

    train_set = ImageDataset(dataset.train, train_transforms)

    if 'triplet' in cfg.DATALOADER.SAMPLER:
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            num_workers=num_workers, collate_fn=train_collate_fn
        )
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    return train_loader, val_loader, len(dataset.query), num_classes, dataset, train_set, train_transforms


def get_trainloader_uda(cfg, trainset=None, num_classes=0):
    if cfg.INPUT.RESIZECROP == True:
        randomcrop = T.RandomResizedCrop(cfg.INPUT.SIZE_TRAIN,scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3)
    else:
        randomcrop = T.RandomCrop(cfg.INPUT.SIZE_TRAIN)
    train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING, padding_mode='constant'),
            #  T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            randomcrop,
            T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.15, saturation=0, hue=0)], p=cfg.INPUT.COLORJIT_PROB),
            T.ToTensor(),
            T.RandomErasing(p=cfg.INPUT.RE_PROB, value=cfg.INPUT.PIXEL_MEAN),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            #RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    train_set = ImageDataset(trainset, train_transforms)

    if 'triplet' in cfg.DATALOADER.SAMPLER:

        if cfg.MODEL.DIST_TRAIN:
            print('DIST_TRAIN START')
            mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // dist.get_world_size()
            data_sampler = RandomIdentitySampler_DDP(trainset, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
            batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
            train_loader = torch.utils.data.DataLoader(
                train_set,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=train_collate_fn,
                pin_memory=True,
            )
        else:
            train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=RandomIdentitySampler(trainset, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=train_collate_fn
            )
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))

    return train_loader


def get_testloader_uda(cfg, aug=False):
    if cfg.INPUT.RESIZECROP == True:
        randomcrop = T.RandomResizedCrop(cfg.INPUT.SIZE_TRAIN,scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3)
    else:
        randomcrop = T.RandomCrop(cfg.INPUT.SIZE_TRAIN)
    if aug == False:
        val_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TEST, interpolation=3),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
        ])
    else:
        val_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TEST, interpolation=3),
            T.RandomHorizontalFlip(p=1.0),
            T.Pad(cfg.INPUT.PADDING, padding_mode='constant'),
            #  T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            randomcrop,
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
        ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR, plus_num_id=cfg.DATASETS.PLUS_NUM_ID)

    val_set = ImageDataset(dataset.gallery, val_transforms)
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    return val_loader, 0, dataset.gallery
