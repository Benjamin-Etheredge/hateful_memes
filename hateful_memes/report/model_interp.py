from captum.attr import KernelShap
from pathlib import Path
import torch
import click
import os
import glob
from icecream import ic
from typing import Tuple
from transformers import BertTokenizer
from torchvision.transforms import PILToTensor, ToPILImage
import cv2

from hateful_memes.data.hateful_memes import MaeMaeDataModule
from hateful_memes.models import(
    VisualBertWithODModule,
    AutoTextModule,
    SimpleImageMaeMaeModel, 
    SimpleMLPImageMaeMaeModel,
    BaseTextMaeMaeModel,
    VisualBertModule,
    BaseITModule
)


def wrapper_forward(image, input_ids, tokenizer_pl_module):
    tokenizer=tokenizer_pl_module[0]
    pl_module=tokenizer_pl_module[1]
    # reassemble dict
    #input_tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist()[0])
    #text_orig = tokenizer.convert_tokens_to_string(input_tokens)
    text_orig = [tokenizer.decode(input_ids.tolist()[0], skip_special_tokens=True)]
    batch = {
        'image': image,
        'text': text_orig
    }
    #ic(batch)
    return pl_module(batch)


def get_attributions(ckpt_dir:str):
    ## DataLoader
    datamodule = MaeMaeDataModule(batch_size=1) # SHAP wants one sample at a time
    datamodule.prepare_data()
    datamodule.setup("fit")
    dataloader = datamodule.train_dataloader()    
    
    ## True model (Just Visual BERT for now)
    assert(Path(ckpt_dir).exists())
    ckpt_search = os.path.join(ckpt_dir, "*.ckpt")
    ckpt_path = glob.glob(ckpt_search)[0]  # Grab the most recent checkpoint?
    pl_module = VisualBertModule.load_from_checkpoint(checkpoint_path=ckpt_path)
    
    ## Calculate feature attribution
    # Features
    data_next = next(iter(dataloader))
    #ic(data_next)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    image = data_next['image']
    label = data_next['label']
    text = data_next['text']
    """token_dict = tokenizer(
        text,
        return_tensors="pt",
        padding='max_length',
        truncation=True,
        max_length=pl_module.max_length)"""
    token_dict = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    ic(text, token_dict)
    # KernelSHAP
    ks = KernelShap(wrapper_forward)
    image_text = (image, token_dict['input_ids'])
    attr = ks.attribute(inputs=image_text, show_progress=True,
        additional_forward_args=[(tokenizer, pl_module)])
    
    return attr


def visualize_attributions(attrs, save_results=False, save_dir="data/08_reporting"):
    ic(attrs)
    if(save_results):
        raise NotImplementedError 


@click.command
@click.option('--ckpt_dir', default='data/06_models/visual_bert')
@click.option('--save_vis', default=False)
@click.option('--save_dir', default='data/08_reporting')
def interp(ckpt_dir:str, save_vis:bool, save_dir:str):
    attrs = get_attributions(ckpt_dir)
    visualize_attributions(attrs, save_vis, save_dir)


if __name__ == '__main__':
    interp()