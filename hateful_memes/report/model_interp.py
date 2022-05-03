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


def rgb_to_grey(rgb):
    grey_coeffs = np.array([[[0.2989, 0.5870, 0.1140]]])
    grey = (rgb * grey_coeffs).sum(axis=2)
    return grey


class InterpModel():
    def __init__(self, model_name:str, ckpt_dir:str):
        assert(Path(ckpt_dir).exists())
        ckpt_search = os.path.join(ckpt_dir, "*.ckpt")
        ckpt_path = glob.glob(ckpt_search)[0]  # Grab the most recent checkpoint?
        self.inner_model = None
        # Input feature attribution parameters
        self.image_embed_layer = None
        self.attr_image_input = None  # defaulted True
        self.text_embed_layer = None
        self.attr_text_input = None  # defaulted False
        self.tokenizer = None
        # Ensemble layer attribution parameters
        self.ensemble_layer = None
        self.attr_ensem_input = None  # default True
        self.sub_models = None
        if model_name == 'visual-bert':
            self.inner_model = VisualBertModule.load_from_checkpoint(checkpoint_path=ckpt_path)
            self.image_embed_layer = self.inner_model.resnet
            self.attr_image_input = True
            self.text_embed_layer = self.inner_model.visual_bert.embeddings.word_embeddings
            self.attr_text_input = False
            self.tokenizer = self.inner_model.tokenizer
        elif model_name == 'beit':
            self.inner_model = BaseITModule.load_from_checkpoint(checkpoint_path=ckpt_path, freeze=False, include_top=True)
            # in_wrap = ModelInputWrapper(self.inner_model.model.beit)
            # self.inner_model.model.beit = in_wrap
            self.image_embed_layer = self.inner_model.model.beit.embeddings.patch_embeddings
            self.attr_image_input = True
            #self.image_embed_layer = in_wrap.input_maps["pixel_values"]
        elif model_name == 'electra':
            self.inner_model = AutoTextModule.load_from_checkpoint(checkpoint_path=ckpt_path)
            self.text_embed_layer = self.inner_model.model.embeddings.word_embeddings
            self.attr_text_input = False
            self.tokenizer = self.inner_model.tokenizer
        elif model_name == 'visual-bert-with-od':
            self.inner_model = VisualBertWithODModule.load_from_checkpoint(checkpoint_path=ckpt_path)
            self.image_embed_layer = self.inner_model.resnet
            self.attr_image_input = True
            self.text_embed_layer = self.inner_model.visual_bert.embeddings.word_embeddings
            self.attr_text_input = False
            self.tokenizer = self.inner_model.tokenizer 
        elif model_name == 'super-model':
            ckpt_storage = os.path.dirname(ckpt_dir)
            self.inner_model = SuperModel.load_from_checkpoint(checkpoint_path=ckpt_path,
                visual_bert_ckpt=os.path.join(ckpt_storage, "visual_bert"),
                #resnet_ckpt=None,
                simple_image_ckpt=os.path.join(ckpt_storage, "simple_image"),
                simple_mlp_image_ckpt=os.path.join(ckpt_storage, "simple_mlp_image"),
                simple_text_ckpt=os.path.join(ckpt_storage, "simple_text"),
                vit_ckpt=os.path.join(ckpt_storage, "vit"),
                beit_ckpt=os.path.join(ckpt_storage, "beit"),
                electra_ckpt=os.path.join(ckpt_storage, "electra"),
                distilbert_ckpt=os.path.join(ckpt_storage, "distilbert")
                #visual_bert_with_od_ckpt=os.path.join(ckpt_storage, "visual_bert_with_od")
                )
            self.ensemble_layer = self.inner_model.fc
            self.sub_models = self.inner_model.models
            self.attr_ensem_input = True
        else:
            raise ValueError("Model named \"%s\" is unsupported." % (model_name))
        self.inner_model.to('cpu')

    # Used as wrapper for model forward()
    def __call__(self, image, input_ids, pil_img_as_tensor, tokenizer):
        t2p = ToPILImage()
        text_orig = [tokenizer.decode(input_id, skip_special_tokens=True)
            for input_id in input_ids.tolist()]
        pil_img = [t2p(tensor) for tensor in pil_img_as_tensor]
        batch = {
            'image': image,
            'text': text_orig,
            'raw_pil_image': pil_img
        }
        return self.inner_model(batch)
    

def get_input_attributions(interp_model:InterpModel, data_sample):
    ## Calculate feature attribution
    # Features
    p2t = PILToTensor()
    image = data_sample['image']
    text = data_sample['text']
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    token_dict = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    token_ids = token_dict['input_ids']
    image_text = (image, token_ids)
    # Feature baselines
    image_baselines = torch.zeros_like(image)
    token_ref= TokenReferenceBase(reference_token_idx=tokenizer.convert_tokens_to_ids("[PAD]"))
    text_baselines = token_ref.generate_reference(token_ids.numel(), device='cpu').unsqueeze(0)
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
                baselines=(image_baselines, text_baselines),
                additional_forward_args=tokenizer,
                attribute_to_layer_input=interp_model.attr_image_input)
            attrs['img'] = img_attr
        if interp_model.text_embed_layer is not None:
            lig_txt = LayerIntegratedGradients(interp_model, interp_model.text_embed_layer)
            txt_attr = lig_txt.attribute(inputs=image_text,
                baselines=(image_baselines, text_baselines),
                additional_forward_args=tokenizer,
                attribute_to_layer_input=interp_model.attr_text_input)
            attrs['txt'] = txt_attr
    elif attr_mode == 'KS': 
        # KernelSHAP
        # Feature masks (?)
        image_mask = torch.tensor(felzenszwalb(image.squeeze(dim=0).numpy(), channel_axis=0,
            scale=0.25, min_size=5))
        text_mask = torch.arange(token_ids.numel()) + image_mask.max()
        # TODO Do we need to convert text to embedding space(?)
        ks = KernelShap(interp_model)
        ks_attr = ks.attribute(image_text, 
            additional_forward_args=tokenizer,
            feature_masks=(image_mask, text_mask),
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
@click.option('--save_name', default='data/08_reporting/interp_data/tmp.pt')
@click.option('--ensemble', is_flag=True, help='Perform model attribution for an ensemble')
def interp(model_name:str, ckpt_dir:str, batch_size:int, save_name:str,
    ensemble:bool):
    
    ## DataLoader
    B = batch_size if ensemble else 1
    datamodule = MaeMaeDataModule(batch_size=B) # Attributors want one sample at a time?
    datamodule.prepare_data()
    datamodule.setup("fit")
    dataloader = datamodule.train_dataloader()    
    data_sample = next(iter(dataloader))

    ## Model to interpret
    interp_model = InterpModel(model_name, ckpt_dir)

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

    torch.save(data_sample, save_name)


if __name__ == '__main__':
    interp()