from typing import Union
from pathlib import Path
from xml.etree.ElementInclude import include
from matplotlib.pyplot import text
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
from transformers import AutoTokenizer, DetrFeatureExtractor, DetrForObjectDetection, AutoFeatureExtractor





def create_vocab_tokenizer(root_dir):
        info = pd.read_json(root_dir/"train.jsonl", lines=True)

class MaeMaeDataset(torch.utils.data.Dataset):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    resizer = T.Resize(size=(224,224))
    od_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    for param in od_model.parameters():
        param.requires_grad = False
    od_model.eval()
    od_feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
    num_queries = 4

    def __init__(
        self, root_dir: str, 
        img_transforms=None,
        txt_transforms=None,
        include_text_features: bool = False,
        pil=False,
        od=False,
        set="train",
        # tokenizer=None,
        # max_length=96,
        # image_feature_extractor=None
    ):
        self.root_dir = Path(root_dir)
        self.img_transforms = img_transforms
        self.txt_transforms = txt_transforms

        if set == "train": 
            self.info = pd.read_json(self.root_dir/"train.jsonl", lines=True)

        elif set == "super_train": 
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
            if "train" in set:
                ic("train set")
                self.img_transforms = self.base_train_img_transforms()
                self.pil_img_transforms = self.base_train_pil_img_transforms()
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
        self.pil = pil
        self.od = od
        # self.tokenizer = tokenizer
        # if tokenizer:
            # self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
            # TODO could just pass through but may be better to make in each thread
        # self.max_length = max_length
        # self.image_feature_extractor = image_feature_extractor
        # if image_feature_extractor:
            # self.image_feature_extractor = AutoFeatureExtractor.from_pretrained(image_feature_extractor)
        # ic(self.od, self.pil, self.tokenizer, self.image_feature_extractor)

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
        # raw_np_image = np.asarray(image)
        if self.img_transforms:
            image = self.img_transforms(image)

        text = data['text']
        if self.txt_transforms:
            text = self.txt_transforms(text)

        label = data['label']
        # TODO maybe make label transformer

        extra_info = {}
        pil_image = self.pil_img_transforms(raw_pil_image)
        # if self.od:
            # Can't do auto because super model
            # extra_info['od_feats'] = self.od_feature_extractor(pil_image, return_tensors='pt')
            # extra_info['image_feats'] = self.image_feature_extractor(pil_image, return_tensors='pt')
            # for key in extra_info['od_image']:
                # ic(key, type(extra_info['od_image'][key]))
            # ic(extra_info['od_image'])
            # extra_info['pil_image'] = pil_image
            # extra_info['pil_image'] = pil_image
        extra_info['pil_image'] = pil_image
            # extra_info['spil_image'] = T.functional.resize(pil_image, (224, 224))
            # image = T.functional.to_tensor(extra_info['spil_image'])
        
        # if self.tokenizer:
            # extra_info['tokenized_text'] = self.tokenizer.encode_plus(text, return_tensors="pt")

        # return (image, text, raw_pil_image, label)
        sample = dict(
            image=image,
            # raw_np_image=raw_np_image,
            # raw_pil_image=raw_pil_image,
            text=text,
            label=label,
            **extra_info
        )

        return sample

    # TODO vocab is broken between train and test
    def base_train_img_transforms(self):
        return T.Compose([
            T.RandomHorizontalFlip(p=0.1),
            # T.RandomVerticalFlip(),
            # transforms.ToPILImage(mode='RGB'),
            # T.RandomRotation(degrees=15),
            T.AutoAugment(T.AutoAugmentPolicy.IMAGENET),
            T.RandomResizedCrop(scale=(0.5, 1), size=(224,224)), # this does good for slowing overfitting
            T.Resize(size=(224,224)),
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
            T.AutoAugment(T.AutoAugmentPolicy.IMAGENET),
            # T.RandomResizedCrop(scale=(0.5, 1), size=(224,224)), # this does good for slowing overfitting
        ])

    def base_test_pil_img_transforms(self):
        return T.Compose([
            # T.Resize(size=(224,224)),
        ])

    @classmethod
    def detect_objects(self, image):

        image_feats = self.od_feature_extractor(images=image, return_tensors="pt")

        image_feats = image_feats
        with torch.no_grad(): 
            od_outputs = self.od_model(**image_feats)
        logits = od_outputs.logits
        probas = logits.softmax(-1)
        batch_keep_idxs = np.argsort(probas.max(-1).values.detach().cpu().numpy())[::-1][:, :self.num_queries]
        # batch_keep_idxs = torch.argsort(torch.max(probas, dim=-1)[0])[::-1][:, :self.num_queries]
        batch_pred_boxes = od_outputs['pred_boxes']

        batch_keep_boxes = []
        for i in range(batch_keep_idxs.shape[0]):
            img_keep_idxs = batch_keep_idxs[i]
            img_pred_boxes = batch_pred_boxes[i]
            img_keep_boxes = img_pred_boxes[img_keep_idxs]
            batch_keep_boxes.append(img_keep_boxes)
        batch_keep_boxes = torch.stack(batch_keep_boxes)

        # crop images
        batch_outputs = []
        batch_inputs = []
        for idx, batch_img in enumerate(image):
            batch_img = T.functional.to_tensor(batch_img)
            w, h = batch_img.shape[2], batch_img.shape[1]
            img_pred_boxes = batch_keep_boxes[idx]
            obj_imgs = []
            for i in range(self.num_queries):
                box = img_pred_boxes[i]
                center_x, center_y, norm_w, norm_h = box
                left = int(max((center_x - norm_w / 2), 0) * w)
                upper = int(max((center_y - norm_h / 2), 0) * h)
                right = int(min((center_x + norm_w / 2), 1) * w)
                lower = int(min((center_y + norm_h / 2), 1) * h)
                # yes, i know this is not a good idea, but it allows us to 
                # handle situations where the object is too small (0 pixels in width or height)
                try:
                    obj_img = batch_img[:, upper:lower, left:right]
                    obj_img = self.normalizer(self.resizer(obj_img))
                except:
                    obj_img = torch.zeros(3, 224, 224)
                obj_imgs.append(obj_img)
            
            # always include full image
            obj_imgs.append(self.normalizer(self.resizer(batch_img)))

            obj_imgs = torch.stack(obj_imgs)
            # obj_img = obj_imgs.to(self.device, non_blocking=True)

            # with torch.no_grad():
            # ic(obj_img.shape)
            # img_outputs = self.resnet(obj_imgs)

            # img_outputs = torch.squeeze(img_outputs)
            # batch_outputs.append(img_outputs)
            batch_inputs.append(obj_img)
        batch_inputs = torch.stack(batch_inputs)
        return batch_inputs

from typing import Dict
def stack_things(things: Dict):
    for key, value in things.items():
        try:
            if type(value[0]) == torch.Tensor:
                # ic('tensor', key)
                things[key] = torch.stack(value)
            # else:
            #     ic("failed", key, type(value[0]))
            #     ic("failed2", type(value[0]))
        except Exception as e:
            # ic("failed tensor", key)
            # ic(e)
            pass
        # else:
            # ic("no match:", key, type(value), type(value[0]))
    return things
 
def create_collate_fn(txt_tok=[], img_tok=[], od=False, pil=False):

    def collate_fn(batch):
        # images, texts, raw_pil_images, labels = zip(*batch)
        # images, texts, raw_pil_images, labels = zip(*batch)

        # ic(batch.keys())
        joined = {key: [] for key in batch[0].keys()}

        # ic(joined.keys())
        for sample in batch:
            for key in sample.keys():
                joined[key].append(sample[key])

        for name, tok in txt_tok:
            tok_txt = tok(joined['text'], return_tensors="pt", truncation=True, padding=True, max_length=96)
            joined[f'{name}'] = stack_things(tok_txt)
            # ic(joined[f'{name}_text'].keys())
            # for k, v in joined[f'{name}_text'].items():
            #     ic(k, type(v))

        if od:
            od_feats = MaeMaeDataset.od_feature_extractor(images=joined['pil_image'], return_tensors="pt")
            joined['od'] = MaeMaeDataset.od_model(**od_feats)

        for name, tok in img_tok:
            joined[f'{name}'] = stack_things(tok(joined['pil_image'], return_tensors="pt"))
            # ic(joined[f'{name}_image'].keys())
            # for k, v in joined[f'{name}_image'].items():
            #     ic(k, type(v))
        # ic(joined.keys())
        # ic(joined['od_feats'])
        # joined['od_feats'] 
        # for k, v in joined.items():
            # ic(k, type(v))
        joined['label'] = torch.tensor(joined['label'])

        if not pil:
            del joined['pil_image']
        
        joined = stack_things(joined)
        return joined

    return collate_fn

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
        img_toks=[],
        txt_toks=[],
        od=False,
        pil=False,
    ):
        super().__init__()
        self.img_toks = img_toks
        self.txt_toks = txt_toks


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
        self.collate_fn = create_collate_fn(self.txt_toks, self.img_toks, od, pil)
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
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            # sampler=torch.utils.data.sampler.WeightedRandomSampler(
                # self.train_dataset.weights, 
                # len(self.train_dataset)//2, # TODO basically 2 gpus does 2 epochs at once without //2
                # replacement=True),
                # (self.train_dataset.num_items//5)*4,
                # replacement=False),
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

