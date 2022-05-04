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

        elif set == "super_train" or set == "val_train": 
            info0 = pd.read_json(self.root_dir/"train.jsonl", lines=True)
            # info1 = pd.read_json(self.root_dir/"dev_seen.jsonl", lines=True)
            info2 = pd.read_json(self.root_dir/"test_seen.jsonl", lines=True)
            self.info = pd.concat([info0, info2])
            # info2 = pd.read_json(self.root_dir/"dev_unseen.jsonl", lines=True)

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
            if "super_train" == set or "train" == set:
                ic("train set")
                self.img_transforms = self.base_train_img_transforms()
                self.pil_img_transforms = self.base_train_pil_img_transforms()
            elif "val_train" == set:
                ic("val train set")
                self.img_transforms = self.base_test_img_transforms()
                self.pil_img_transforms = self.base_test_pil_img_transforms()
            else:
                ic("test set")
                self.img_transforms = self.base_test_img_transforms()
                self.pil_img_transforms = self.base_test_pil_img_transforms()

        self.include_text_features = include_text_features
        self.class_weights = self.info['label'].value_counts().to_numpy()
        ic(self.class_weights)
        self.class_weights = self.class_weights / self.class_weights.sum()
        self.class_weights = 1 / self.class_weights
        ic(self.class_weights)
        self.weights = [self.class_weights[0] if label==0 else self.class_weights[1] for label in self.info['label']]
        # self.weights = np.where(self.info['label'].to_numpy(), self.class_weights)
        self.weights = torch.tensor(self.weights)

        self.vocab = None

    def __len__(self):
        return len(self.info)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = self.info.iloc[idx]
        img_path = self.root_dir/data['img']
        # image = io.imread(img_path)
        image = Image.open(img_path).convert('RGB')
        # image = self.images[idx]
        raw_pil_image = image
        if self.pil_img_transforms is not None:
            raw_pil_image = self.pil_img_transforms(image)
        # raw_np_image = np.asarray(image)
        if self.img_transforms:
            image = self.img_transforms(image)

        text = data['text']
        if self.txt_transforms:
            text = self.txt_transforms(text)

        label = data['label']
        # TODO maybe make label transformer
        img_id = data['id']

        extra_text_info = {}

        return (image, text, raw_pil_image, label)


    # TODO vocab is broken between train and test
    def base_train_img_transforms(self):
        return T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.1),
            # transforms.ToPILImage(mode='RGB'),
            T.RandomRotation(degrees=15),
            T.AutoAugment(T.AutoAugmentPolicy.IMAGENET),
            T.RandomResizedCrop(scale=(0.2, 1), size=(224,224)), # this does good for slowing overfitting
            # T.Resize(size=(224,224)),
            T.ToTensor(), # this already seems to scale okay
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            # T.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=True),
            T.RandomErasing(),
        ])

    def base_test_img_transforms(self):
        return T.Compose([
            # T.RandomHorizontalFlip(),
            # transforms.ToPILImage(mode='RGB'),
            T.Resize(size=(224,224)),
            T.ToTensor(), # this already seems to scale okay
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
    def base_train_pil_img_transforms(self):
        return T.Compose([
            T.RandomHorizontalFlip(p=0.1),
            T.RandomVerticalFlip(p=0.1),
            T.RandomRotation(degrees=15),
            T.AutoAugment(T.AutoAugmentPolicy.IMAGENET),
            # T.RandomResizedCrop(scale=(0.5, 1), size=(224,224)), # this does good for slowing overfitting
        ])

    def base_test_pil_img_transforms(self):
        return T.Compose([
            # T.Resize(size=(224,224)),
        ])

def collate_fn(batch):
    images, texts, raw_pil_images, labels = zip(*batch)
    # images, texts, raw_pil_images, labels = zip(*batch)

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
        raw_pil_image=list(raw_pil_images),
        # raw_np_image=raw_np_images,,
        text=list(texts),
        label=labels,
    )


class MaeMaeDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        data_dir: str = './data/01_raw/hateful_memes',
        batch_size: int = 128,
        img_transforms=None, 
        txt_transforms=None,
        num_workers=None,
        pin_memory=False,
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
            num_workers = max(1, min(os.cpu_count()//2, 12))
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
        if stage=="validate":
            self.train_dataset = MaeMaeDataset(
                self.data_dir,
                img_transforms=self.img_transforms, 
                txt_transforms=self.txt_transforms,
                set="val_train",
            )
        else:
            self.train_dataset = MaeMaeDataset(
                self.data_dir,
                img_transforms=self.img_transforms, 
                txt_transforms=self.txt_transforms,
                set="super_train",
            )
        self.val_dataset = MaeMaeDataset(
            self.data_dir,
            img_transforms=self.img_transforms, 
            txt_transforms=self.txt_transforms,
            set='dev_unseen',
        )
        self.test_dataset = MaeMaeDataset(
            self.data_dir,
            img_transforms=self.img_transforms, 
            txt_transforms=self.txt_transforms,
            set="test_unseen",
        )

        train_ids = set(self.train_dataset.info['id'])
        val_ids = set(self.val_dataset.info['id'])
        test_ids = set(self.test_dataset.info['id'])
        assert len(train_ids.intersection(val_ids)) == 0
        assert len(train_ids.intersection(test_ids)) == 0
        assert len(val_ids.intersection(test_ids)) == 0

    def train_dataloader(self, shuffle=True, drop_last=True):
        if shuffle:
            kwargs = dict(
                sampler=torch.utils.data.sampler.WeightedRandomSampler(
                    self.train_dataset.weights, 
                    len(self.train_dataset), # TODO basically 2 gpus does 2 epochs at once without //2
                    replacement=True,
                )
            )
            # kwargs = dict(shuffle=True)
        else:
            kwargs = dict(shuffle=False)

        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            **kwargs,
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
            num_workers=4,
            pin_memory=self.pin_memory,
            persistent_workers=self.persitent_workers,
            collate_fn=self.collate_fn,
        )
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=self.pin_memory,
            persistent_workers=self.persitent_workers,
            collate_fn=self.collate_fn,
        ) 

