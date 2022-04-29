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
from skimage.segmentation import felzenszwalb 
from skimage.color import rgb2gray
from captum.attr import visualization, TokenReferenceBase
import matplotlib.pyplot as plt
import numpy as np

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
    image = data_next['image']
    text = data_next['text']
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    token_dict = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    token_ids = token_dict['input_ids']
    # Feature masks
    image_mask = torch.tensor(felzenszwalb(image.squeeze(dim=0).numpy(), channel_axis=0,
        scale=0.5, min_size=10))
    text_mask = torch.arange(token_ids.numel()) + image_mask.max()
    # Feature baselines
    image_baselines = torch.zeros_like(image)
    token_ref= TokenReferenceBase(reference_token_idx=tokenizer.convert_tokens_to_ids("[PAD]"))
    text_feature_baselines = token_ref.generate_reference(token_ids.numel(), device='cpu')
    # KernelSHAP
    ks = KernelShap(wrapper_forward)
    image_text = (image, token_ids)
    img_attr, txt_attr = ks.attribute(inputs=image_text, show_progress=True,
        feature_mask=(image_mask, text_mask),
        baselines=(image_baselines, text_feature_baselines),
        additional_forward_args=[(tokenizer, pl_module)], n_samples=100)

    ## Outputs
    attrs = {
        'img':img_attr,
        'txt':txt_attr
    } 
    inputs = {
        'img':image,
        'txt': text
    }
    return attrs, inputs


def visualize_attributions(attrs, inputs, save_results=False, save_dir="data/08_reporting"):
    if 'img' in attrs.keys():
        img_attr = attrs['img']
        img_in = inputs['img']
        img_attr_vis = img_attr.squeeze(dim=0).permute(1, 2, 0).numpy()
        img_in_vis = img_in.squeeze(dim=0).permute(1, 2, 0).numpy()
        fig, ax = visualization.visualize_image_attr(img_attr_vis, original_image=img_in_vis,
            method="blended_heat_map", sign="all", show_colorbar=True)
        if(save_results):
           fig.savefig(os.path.join(save_dir, "img_attr.png"))

    if 'txt' in attrs.keys():
        # Look at https://captum.ai/tutorials/Multimodal_VQA_Interpret
        # something = visualization.VisualizationDataRecord(...)
        txt_attr = attrs['txt']
        ic(txt_attr)
   

@click.command
@click.option('--ckpt_dir', default='data/06_models/visual_bert')
@click.option('--save_vis', is_flag=True)
@click.option('--save_dir', default='data/08_reporting')
def interp(ckpt_dir:str, save_vis:bool, save_dir:str):
    attrs, inputs = get_attributions(ckpt_dir)
    visualize_attributions(attrs, inputs, save_vis, save_dir)


if __name__ == '__main__':
    interp()