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


        # if self.txt_transforms is None:
        #     self.txt_transforms = self.create_text_transform()
        if self.img_transforms is None:
            self.img_transforms = self.base_img_transforms()

        self.include_text_features = include_text_features
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
        if self.img_transforms:
            image = self.img_transforms(image)

        text = data['text']
        if self.txt_transforms:
            text = self.txt_transforms(text)

        label = data['label']
        # TODO maybe make label transformer

        extra_text_info = {}

        sample = dict(
            image=image,
            text=text,
            label=label,
            **extra_text_info
        )

        return sample

    # TODO vocab is broken between train and test
    def base_img_transforms(self):
        return T.Compose([
            # T.RandomHorizontalFlip(),
            # transforms.ToPILImage(mode='RGB'),
            T.Resize(size=(224,224)),
            T.ToTensor(), # this already seems to scale okay
            T.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ])


    def create_text_transform(self):
        # TODO maybe move into dataset
        from torchtext.data.utils import get_tokenizer
        from torchtext.vocab import build_vocab_from_iterator

        self.tokenizer = get_tokenizer('basic_english')
        def yield_tokens(data_iter):
            pool = Pool(processes=os.cpu_count())
            # return pool.map(self.tokenizer, [item['text'] for item in data_iter])
            return pool.map(self.tokenizer, self.info['text'])
            # for item in data_iter:
            #     # ic(item)
            #     yield tokenizer(item['text'])
        self.vocab = build_vocab_from_iterator(yield_tokens(None), specials=["<unk>"])
        self.vocab.set_default_index(self.vocab["<unk>"])

        return lambda x: self.vocab(self.tokenizer(x))

class MaeMaeDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        data_dir: str = './data/01_raw/hateful_memes',
        batch_size: int = 128,
        img_transforms=None, 
        txt_transforms=None,
        num_workers=None,
        pin_memory=False,
        collate_fn=None,
        # tokenizer=None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        # self.img_transforms = img_transforms
        self.img_transforms = img_transforms
        self.txt_transforms = txt_transforms

        if num_workers is None:
            num_workers = max(1, min(os.cpu_count()//2, 8))
        self.num_workers = num_workers

        self.pin_memory = pin_memory

        self.tokenizer = None
        self.vocab = None
        self.collate_fn = collate_fn
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

    
    # def create_collate_fn(self):
        # if self.collate_fn is None:
        #     return default_collate
        # else:
        #     return self.collate_fn

    def train_dataloader(self, shuffle=True, drop_last=True):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=drop_last,
            persistent_workers=False,
            # collate_fn=MaeMaeDataset.collate_batch,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=False,
            # collate_fn=MaeMaeDataset.collate_batch,
            # collate_fn=self.collate_batch, = None
        )
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.test_num_workers,
            pin_memory=self.pin_memory,
            # collate_fn=MaeMaeDataset.collate_batch,
            # collate_fn=self.collate_batch,
        ) 

    # def collate_batch(self, batch):
    #     img_list, labels_list, raw_text, text_list, offsets = [], [], [], [], [0]
    #     # ic(batch)
    #     # if type(batch) == list:
    #         # ic(batch)
    #     for item in batch:
    #         img_list.append(item['image'])
    #         labels_list.append(item['label'])

    #         _text = item['text']
    #         processed_text = torch.tensor(self.text_pipeline(_text), dtype=torch.int64)
    #         raw_text.append(_text)
    #         text_list.append(processed_text)
    #         offsets.append(processed_text.size(0))

    #     img_list = torch.torch(img_list)
        
    #     labels_list = torch.tensor(labels_list, dtype=torch.int32)
    #     offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    #     text_list = torch.cat(text_list)
    #     return dict(
    #         text_features=text_list,
    #         text_offset=offsets,
    #         label=labels_list,
    #         image=img_list,
    #         text=raw_text,
    #     )
            
# def create_transformer(img_transforms=None, text_transforms=None):
#     def wrapper_transformer(sample: Dict):
#         if img_transforms:
#             sample['image'] = img_transforms(sample['image'])
#         if text_transforms:
#             sample['text'] = text_transforms(sample['text'])
#         return sample
#     return wrapper_transformer

