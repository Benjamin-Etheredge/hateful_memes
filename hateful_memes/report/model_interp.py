from captum.attr import KernelShap, LayerDeepLiftShap, LayerIntegratedGradients
from captum.attr._utils.input_layer_wrapper import ModelInputWrapper
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
from captum.attr import (
        visualization, 
        TokenReferenceBase, 
        configure_interpretable_embedding_layer, 
        remove_interpretable_embedding_layer
    )
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
import numpy as np
import sys
import torch.nn.functional as F
from math import ceil, floor
import json
from pytorch_lightning.utilities.seed import seed_everything

import hateful_memes
from hateful_memes.data.hateful_memes import MaeMaeDataModule
from hateful_memes.models.visual_bert_with_od import VisualBertWithODModule
from hateful_memes.models.auto_text_model import AutoTextModule
from hateful_memes.models.simple_image import SimpleImageMaeMaeModel
from hateful_memes.models.simple_mlp_image import SimpleMLPImageMaeMaeModel
from hateful_memes.models.simple_text import BaseTextMaeMaeModel
from hateful_memes.models.visual_bert import VisualBertModule
from hateful_memes.models.baseIT import BaseITModule
from hateful_memes.models.super_model import SuperModel
from hateful_memes.utils import get_checkpoint_filename
from hateful_memes.report.model_wrappers import InterpModel





def get_input_attributions(interp_model:InterpModel, data_sample):
    ## Calculate feature attribution
    # Features
    p2t = PILToTensor()
    image = data_sample['image']  # tensor with shape (1, C, H, W)
    text = data_sample['text']  # tuple of 1 strings
    raw_pil = data_sample['raw_pil_image']  # tuple of 1 PIL Image objects
    pil_image = p2t(raw_pil[0]).unsqueeze(0)  # tensor with shape (1, C, H, W)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    token_dict = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    token_ids = token_dict['input_ids']
    image_text = (image, token_ids, pil_image)
    # Feature baselines
    image_baselines = torch.zeros_like(image)
    token_ref= TokenReferenceBase(reference_token_idx=tokenizer.convert_tokens_to_ids("[PAD]"))
    text_baselines = token_ref.generate_reference(token_ids.numel(), device='cpu').unsqueeze(0)
    pil_baselines = torch.zeros_like(pil_image)
    # Prep for attribution
    attrs = {}
    interp_model.inner_model.eval()
    attr_mode = 'IG' 
    if attr_mode == 'IG':
        # Layer Integrated Gradients
        if interp_model.image_embed_layer is None and interp_model.text_embed_layer is None:
            raise RuntimeError("Interpretable Model is missing an input feature layer.")
        if interp_model.image_embed_layer is not None:
            lig_image = LayerIntegratedGradients(interp_model, interp_model.image_embed_layer)
            img_attr = lig_image.attribute(inputs=image_text,
                baselines=(image_baselines, text_baselines, pil_baselines),
                additional_forward_args=tokenizer,
                attribute_to_layer_input=interp_model.attr_image_input)
            attrs['img'] = img_attr
        if interp_model.text_embed_layer is not None:
            lig_txt = LayerIntegratedGradients(interp_model, interp_model.text_embed_layer)
            txt_attr = lig_txt.attribute(inputs=image_text,
                baselines=(image_baselines, text_baselines, pil_baselines),
                additional_forward_args=tokenizer,
                attribute_to_layer_input=interp_model.attr_text_input)
            attrs['txt'] = txt_attr
    elif attr_mode == 'KS': 
        # KernelSHAP
        # Feature masks (?)
        image_mask = torch.tensor(felzenszwalb(image.squeeze(dim=0).numpy(), channel_axis=0,
            scale=0.25, min_size=5))
        text_mask = torch.arange(token_ids.numel()) + image_mask.max()
        pil_mask = torch.tensor(felzenszwalb(pil_image.squeeze(dim=0).numpy(), channel_axis=0,
            scale=0.25, min_size=5)) 
        # TODO Do we need to convert text to embedding space(?)
        ks = KernelShap(interp_model)
        ks_attr = ks.attribute(inputs=image_text, 
            additional_forward_args=[tokenizer,],
            feature_mask=(image_mask, text_mask, pil_mask),
            baselines=(image_baselines, text_baselines, pil_baselines),
            show_progress=True)   
        attrs['img'] = ks_attr[0]
        attrs['txt'] = ks_attr[1] 
    
    return attrs


def get_ensemble_attributions(interp_model:InterpModel, data_sample):
    ## Calculate feature attribution
    # Features
    p2t = PILToTensor()
    images = data_sample['image']  # tensor with shape (N, C, H, W)
    texts = data_sample['text']  # tuple of N strings
    raw_pils = data_sample['raw_pil_image']  # tuple of N PIL Image objects
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    attrs = {}
    interp_model.inner_model.eval()
    mod_attr = None
    for i in range(images.shape[0]):
        print("Attribution %i of %i..." % (i+1, images.shape[0]))
        image = images[i].unsqueeze(0)  # tensor with shape (1, C, H, W)
        text = [texts[i]]  # list with one string
        pil_image = p2t(raw_pils[i]).unsqueeze(0)  # tensor with shape (1, C, H, W)
        token_dict = tokenizer(text, padding=True, return_tensors="pt")
        token_ids = token_dict['input_ids']
        image_text = (image, token_ids, pil_image)
        # Feature baselines
        image_baselines = torch.zeros_like(image)
        token_ref= TokenReferenceBase(reference_token_idx=tokenizer.convert_tokens_to_ids("[PAD]"))
        text_baselines = token_ref.generate_reference(token_ids.numel(), device='cpu').unsqueeze(0)
        pil_baselines = torch.zeros_like(pil_image)
        # Layer Integrated Gradients
        if interp_model.ensemble_layer is None:
            raise RuntimeError("Interpretable Model is missing an Ensemble Layer")
        lig_mod = LayerIntegratedGradients(interp_model, interp_model.ensemble_layer)
        mod_attr_i = lig_mod.attribute(inputs=image_text,
            baselines=(image_baselines, text_baselines, pil_baselines),
            additional_forward_args=tokenizer,
            attribute_to_layer_input=interp_model.attr_ensem_input)
        if mod_attr is None:
            mod_attr = mod_attr_i
        else:
            mod_attr = torch.cat((mod_attr, mod_attr_i))  

    attrs['models'] = mod_attr
    return attrs


@click.command
@click.option('--model_name', default='visual-bert')
@click.option('--ckpt_dir', default='data/06_models/visual_bert')
@click.option('--batch_size', default=1, help='Batch size for ensemble-level attribution')
@click.option('--save_dir', default='data/08_reporting')
@click.option('--save_prefix', default='tmp')
@click.option('--ensemble', is_flag=True, help='Perform model attribution for an ensemble')
@click.option('--trials', default=1, help='Number of times to call interp()')
def interp(model_name:str, ckpt_dir:str, batch_size:int, save_dir:str, save_prefix:str,
    ensemble:bool, trials:int):
    
    seed_everything(42)

    ## DataLoader
    B = batch_size if ensemble else 1
    datamodule = MaeMaeDataModule(batch_size=B) # Attributors want one sample at a time?
    datamodule.prepare_data()
    datamodule.setup("fit")

    dataloader = torch.utils.data.DataLoader(
        datamodule.test_dataset,
        batch_size=datamodule.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=datamodule.pin_memory,
        persistent_workers=datamodule.persitent_workers,
        collate_fn=datamodule.collate_fn,
    ) 
    
    ## Set up outdir
    if not Path(save_dir).exists():
        os.mkdir(save_dir)

    ## Model to interpret
    interp_model = InterpModel(model_name, ckpt_dir)


    for t, data_sample in enumerate(dataloader):
        if t >= trials:
            break

        ## Get attributions
        attrs = None
        if ensemble:
            attrs = get_ensemble_attributions(interp_model, data_sample)
        else:
            attrs = get_input_attributions(interp_model, data_sample)

        ## Save for visualization
        data_sample.update(attrs)
        util_entries = {
            'model_name':model_name,
            'ckpt_dir':ckpt_dir
        }
        data_sample.update(util_entries)

        ic(data_sample['text'])

        trial_save = save_prefix + "_%i.pt" % (t,)
        trial_path = os.path.join(save_dir, trial_save)
        torch.save(data_sample, trial_path)


if __name__ == '__main__':
    interp()