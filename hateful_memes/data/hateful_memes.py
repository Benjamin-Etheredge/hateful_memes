from typing import Union
from pathlib import Path
from xml.etree.ElementInclude import include
import pandas as pd
import torch
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True # TODO 
import pytorch_lightning as pl
from torch.utils.data import random_split
from typing import Dict
from icecream import ic
from multiprocessing import Pool
import os
import transformers
from torchvision import transforms as T
import numpy as np


def create_vocab_tokenizer(root_dir):
        info = pd.read_json(root_dir/"train.jsonl", lines=True)

class MaeMaeDataset(torch.utils.data.Dataset):
    def __init__(
        self, root_dir: str, 
        img_transforms=None,
        txt_transforms=None,
        include_text_features: bool = False,
        set="train",
    ):
        self.root_dir = Path(root_dir)
        self.img_transforms = img_transforms
        self.txt_transforms = txt_transforms

        if set == "train": 
            self.info = pd.read_json(self.root_dir/"train.jsonl", lines=True)
        elif set == "dev_seen":
            self.info = pd.read_json(self.root_dir/"dev_seen.jsonl", lines=True)
        elif set == "dev_unseen":
            self.info = pd.read_json(self.root_dir/"dev_unseen.jsonl", lines=True)
        elif set == "dev":
            info1 = pd.read_json(self.root_dir/"dev_seen.jsonl", lines=True)
            info2 = pd.read_json(self.root_dir/"dev_unseen.jsonl", lines=True)
            self.info = pd.concat([info1, info2])
        elif set == "test_seen":
            self.info = pd.read_json(self.root_dir/"test_seen.jsonl", lines=True)
        elif set == "test_unseen":
            self.info = pd.read_json(self.root_dir/"test_unseen.jsonl", lines=True)
        else:
            raise ValueError(f"Unknown set: {set}")

        # self.images = [
        #     Image.open(self.root_dir/path).convert('RGB') 
        #     for path in self.info['img']
        # ]
        # self.texts = self.info['text']
        # self.labels = self.info['label']


        # if self.txt_transforms is None:
        #     self.txt_transforms = self.create_text_transform()
        if self.img_transforms is None:
            if set == "train":
                self.img_transforms = self.base_train_img_transforms()
            else:
                self.img_transforms = self.base_test_img_transforms()

        self.include_text_features = include_text_features
        self.vocab = None
        self.vit_T = None

    def __len__(self):
        return len(self.info)

    @staticmethod
    def vit(x):
        return self.vit_T(x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = self.info.iloc[idx]
        img_path = self.root_dir/data['img']
        # image = io.imread(img_path)
        image = Image.open(img_path).convert('RGB')
        # image = self.images[idx]
        raw_pil_image = image.resize((224, 224))
        # raw_np_image = np.asarray(image)
        if self.img_transforms:
            image = self.img_transforms(image)

        text = data['text']
        if self.txt_transforms:
            text = self.txt_transforms(text)

        label = data['label']
        # TODO maybe make label transformer

        extra_text_info = {}

        return (image, text, raw_pil_image, label)
        sample = dict(
            image=image,
            # raw_np_image=raw_np_image,
            # raw_pil_image=raw_pil_image,
            text=text,
            label=label,
            **extra_text_info
        )

        return sample

    # TODO vocab is broken between train and test
    def base_train_img_transforms(self):
        return T.Compose([
            T.RandomHorizontalFlip(),
            # T.RandomVerticalFlip(),
            # transforms.ToPILImage(mode='RGB'),
            # T.RandomResizedCrop(size=(224,224)),
            # T.RandomRotation(degrees=15),
            T.Resize(size=(224,224)),
            T.ToTensor(), # this already seems to scale okay
            T.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
            T.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=True),
        ])

    def base_test_img_transforms(self):
        return T.Compose([
            # T.RandomHorizontalFlip(),
            # transforms.ToPILImage(mode='RGB'),
            T.Resize(size=(224,224)),
            T.ToTensor(), # this already seems to scale okay
            # T.Normalize(mean=[0.485, 0.456, 0.406],
            #                       std=[0.229, 0.224, 0.225]),
        ])

def collate_fn(batch):
    images, texts, raw_pil_images, labels = zip(*batch)
    # images, texts, raw_pil_images, labels = zip(*batch)

    # for sample in batch:
    #     images.append(sample['image'])
    #     raw_pil_images.append(sample['raw_pil_image'])
    #     # raw_np_images.append(sample['raw_np_image'])
    #     texts.append(sample['text'])
    #     labels.append(sample['label'])
    
    images = torch.stack(images, dim=0)
    labels = torch.as_tensor(labels)
    return dict(
        image=images,
        raw_pil_image=raw_pil_images,
        # raw_np_image=raw_np_images,
        text=texts,
        label=labels
    )


class MaeMaeDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        data_dir: str = './data/01_raw/hateful_memes',
        batch_size: int = 128,
        img_transforms=None, 
        txt_transforms=None,
        num_workers=None,
        pin_memory=True,
        persistent_workers=True,
        # collate_fn=None,
        # tokenizer=None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_transforms = img_transforms
        self.txt_transforms = txt_transforms

        if num_workers is None:
            num_workers = max(1, min(os.cpu_count()//2, 8))
        self.num_workers = num_workers
        ic(self.num_workers)

        self.pin_memory = pin_memory
        self.persitent_workers = persistent_workers

        self.tokenizer = None
        self.vocab = None
        self.collate_fn = collate_fn
        # self.collate_fn = None
        self.save_hyperparameters()
    
    def prepare_data(self) -> None:
        return super().prepare_data()
    
    def setup(self, stage: str):
        # self.log(batch_size=self.batch_size)
        self.train_dataset = MaeMaeDataset(
            self.data_dir,
            img_transforms=self.img_transforms, 
            txt_transforms=self.txt_transforms,
            set="train",
        )
        self.val_dataset = MaeMaeDataset(
            self.data_dir,
            img_transforms=self.img_transforms, 
            txt_transforms=self.txt_transforms,
            set='dev_seen',
        )
        self.test_dataset = MaeMaeDataset(
            self.data_dir,
            img_transforms=self.img_transforms, 
            txt_transforms=self.txt_transforms,
            set="test_seen",
        )

    def train_dataloader(self, shuffle=True, drop_last=True):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persitent_workers,
            drop_last=drop_last,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persitent_workers,
            collate_fn=self.collate_fn,
        )
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.test_num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persitent_workers,
            collate_fn=self.collate_fn,
        ) 

