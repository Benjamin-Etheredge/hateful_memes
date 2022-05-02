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
                distilbert_ckpt=os.path.join(ckpt_storage, "distilbert"),
                visual_bert_with_od_ckpt=os.path.join(ckpt_storage, "visual_bert_with_od")
                )
            self.ensemble_layer = self.inner_model.dense_model
            self.sub_models = self.inner_model.models
            self.attr_ensem_input = True
        else:
            raise ValueError("Model named \"%s\" is unsupported." % (model_name))
        self.inner_model.to('cpu')

    # Used as wrapper for model forward()
    def __call__(self, image, input_ids, tokenizer):
        text_orig = [tokenizer.decode(input_id, skip_special_tokens=True)
            for input_id in input_ids.tolist()]
        batch = {
            'image': image,
            'text': text_orig
        }
        return self.inner_model(batch)
    

def get_input_attributions(interp_model:InterpModel, data_sample):
    ## Calculate feature attribution
    # Features
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
        # Superpixel feature mask for image
        # Convert text to embedding space

        ks = KernelShap(interp_model)
        ks_attr = ks.attribute(image_text, 
            additional_forward_args=tokenizer,
            #feature_masks=,
            show_progress=True)   
        attrs['img'] = ks_attr[0]
        attrs['txt'] = ks_attr[1] 
    
    return attrs


def get_model_attributions(interp_model:InterpModel, data_sample):
    ## Calculate feature attribution
    # Features
    images = data_sample['image']
    texts = data_sample['text']
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    attrs = {}
    interp_model.inner_model.eval()
    mod_attr = None
    for i in range(images.shape[0]):
        print("Attribution %i of %i..." % (i+1, images.shape[0]))
        image = images[i].unsqueeze(0)
        text = [texts[i]]    
        token_dict = tokenizer(text, padding=True, return_tensors="pt")
        token_ids = token_dict['input_ids']
        image_text = (image, token_ids)
        # Feature baselines
        image_baselines = torch.zeros_like(image)
        token_ref= TokenReferenceBase(reference_token_idx=tokenizer.convert_tokens_to_ids("[PAD]"))
        text_baselines = token_ref.generate_reference(token_ids.numel(), device='cpu').unsqueeze(0)
        # Layer Integrated Gradients
        if interp_model.ensemble_layer is None:
            raise RuntimeError("Interpretable Model is missing an Ensemble Layer")
        lig_mod = LayerIntegratedGradients(interp_model, interp_model.ensemble_layer)
        mod_attr_i = lig_mod.attribute(inputs=image_text,
            baselines=(image_baselines, text_baselines),
            additional_forward_args=tokenizer,
            attribute_to_layer_input=interp_model.attr_ensem_input)
        if mod_attr is None:
            mod_attr = mod_attr_i
        else:
            mod_attr = torch.cat((mod_attr, mod_attr_i))  

    attrs['models'] = mod_attr
    return attrs


def visualize_input_attributions(attrs, inputs, y_hat, y, tokenizer, model_name,
    save_dir="data/08_reporting", save_name="tmp.png"):
    
    # Prediction string
    y_hat = y_hat.item()
    pred = 1 if y_hat>0.5 else 0
    y_hat_label = "Hateful" if pred==1 else "Not Hateful" 
    y_label = "Hateful" if y==1 else "Not Hateful" 
    pn_str = "positive"
    tf_str = "false"
    if y==pred:
        tf_str = "true"
    if pred==0:
        pn_str = "negative"
    pred_str = "Model predicted: \"%s\", w/ logit %.3g\n(%s %s)" % (y_hat_label, y_hat, tf_str, pn_str)

    # Both visualizations use the original image
    img_in = inputs['img']
    img_in_vis = img_in.squeeze(dim=0).permute(1, 2, 0).numpy()

    # Set figure up 
    num_subs = 1
    width_ratios=[5]
    if 'img' in attrs.keys():
        num_subs += 1
        width_ratios.append(6)
    if 'txt' in attrs.keys():
        num_subs += 3
        width_ratios.append(1.5)
        width_ratios.append(2)
        width_ratios.append(1.5)
    if 'txt' in attrs.keys() and 'img' in attrs.keys():
        num_subs += 3
        width_ratios.append(1.5)
        width_ratios.append(2)
        width_ratios.append(1.5)
    width_required = ceil(np.sum(width_ratios) * (12./20.)) + 2      
    fig = plt.figure(figsize=(width_required, 5), constrained_layout=False)
    gs = fig.add_gridspec(1, num_subs, left=0.025, right=1.0, bottom=0.1, top=0.75, wspace=0.1, width_ratios=width_ratios)
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_orig.imshow(img_in_vis)
    ax_orig.xaxis.set_ticks_position("none")
    ax_orig.yaxis.set_ticks_position("none")
    ax_orig.set_xticklabels([])
    ax_orig.set_yticklabels([])
    ax_orig.set_title("Original")
    current_sub = 1

    img_attr = None 
    img_attr_comb = None
    txt_attr = None
    all_attrs = None
    
    if 'img' in attrs.keys():
        img_attr = attrs['img']
        img_attr_vis = img_attr.squeeze(dim=0).permute(1, 2, 0).numpy()
        img_attr_comb = np.sum(img_attr_vis, axis=-1)
        all_attrs = img_attr_comb.copy().flatten()

    if 'txt' in attrs.keys():
        txt_attr = attrs['txt'].squeeze(0)  # Should now be (max_length x embed_dim)
        txt_attr = txt_attr.sum(-1)
        if all_attrs is not None:
            all_attrs = np.concatenate([all_attrs, txt_attr.numpy().copy().flatten()])
        else:
            all_attrs = np.concatenate(txt_attr.numpy().copy().flatten())

    attrs_norm = np.linalg.norm(all_attrs)

    # Visualize image feature attributions
    if 'img' in attrs.keys():
        axi = fig.add_subplot(gs[0, current_sub])
        current_sub += 1
        img_attr_comb = np.sum(img_attr_vis, axis=-1)
        #img_attr_normed = img_attr_comb/np.linalg.norm(img_attr_comb)
        img_attr_normed = img_attr_comb/attrs_norm
        grey_img = rgb_to_grey(img_in_vis)
        imi0 = axi.imshow(grey_img, cmap=plt.get_cmap('gray'))
        imi = axi.imshow(img_attr_normed, alpha=0.7, cmap='bwr')
        axi.xaxis.set_ticks_position("none")
        axi.yaxis.set_ticks_position("none")
        axi.set_xticklabels([])
        axi.set_yticklabels([])        
        axi.set_title("Norm. Visual Attr.")
        fig.colorbar(imi, ax=axi)
    
    # Visualize text feature attributions
    if 'txt' in attrs.keys():
        #txt_attr = attrs['txt'].squeeze(0)  # Should now be (max_length x embed_dim)
        txt_in = inputs['txt']
        token_dict = tokenizer(
            txt_in, 
            return_tensors="pt", 
            padding='max_length', 
            truncation=True, 
            max_length=txt_attr.shape[0],
            add_special_tokens=True, return_special_tokens_mask=True)
        
        token_mask = token_dict['special_tokens_mask'] < 1
        #txt_attr = txt_attr.sum(-1)
        txt_attr = torch.masked_select(txt_attr, token_mask.squeeze(0))
        txt_attr_norm = txt_attr.norm()
        #txt_attr_normed = txt_attr/txt_attr_norm
        txt_attr_normed = txt_attr/attrs_norm
        
        axt = fig.add_subplot(gs[0, current_sub+1])
        current_sub += 3
        imt = axt.imshow(txt_attr_normed.unsqueeze(-1).numpy(), cmap='bwr')
        txt_ids_masked = torch.masked_select(torch.tensor(token_dict['input_ids']), 
            token_mask.squeeze(0)).tolist()
        txt_labels = [tokenizer.decode(id) for id in txt_ids_masked]
        axt.set_yticks(np.arange(txt_attr_normed.numel()), labels=txt_labels)
        #for idx, attr_i in enumerate(txt_attr_normed.tolist()):
        #    axt.text(0, idx, '%.2g' % (attr_i,), ha="center", va="center", color="w")
        fig.colorbar(imt, ax=axt)
        axt.xaxis.set_ticks_position("none")
        axt.set_xticklabels([])
        axt.set_title("Norm. Text Attr.")

    if 'txt' in attrs.keys() and 'img' in attrs.keys():
        total_abs_vis = img_attr.sum().item()
        total_abs_txt = txt_attr.sum().item()
        axb = fig.add_subplot(gs[0, current_sub+1])
        axb.bar(["Visual", "Text"], [total_abs_vis, total_abs_txt])
        axb.set_title("Net Total Attribution")

    fig.suptitle("%s Attribution Scores\n%s" % (model_name, pred_str))
    fig.savefig(os.path.join(save_dir, save_name))
       

def visualize_model_attributions(attrs, inputs, y_hat, y, sub_models, model_name, 
    save_dir="data/08_reporting", save_name="tmp.png"):
    ensem_attr = attrs['models']
    ensem_attr_mean = ensem_attr.mean(dim=0)
    ensem_attr_normed = ensem_attr_mean/ensem_attr_mean.norm()
    # Get sub-model output sizes
    hidden_size = [sub.last_hidden_size for sub in sub_models]
    # Populate sub-model attributions
    sub_attr = {}
    sub_attr_normed = {}
    sub_attr_total = {}
    attr_start = 0
    max_hidden = -sys.maxsize
    min_hidden = sys.maxsize
    max_attr = -sys.maxsize
    min_attr = sys.maxsize
    for i, sub in enumerate(sub_models):
        sub_mod_name = sub.__class__.__name__ + ("(%i)"%(i+1,))
        attr_stop = attr_start + hidden_size[i]
        this_ensem_attr = ensem_attr[:, attr_start:attr_stop]
        sub_attr[sub_mod_name] = this_ensem_attr        
        this_ensem_attr_normed = ensem_attr_normed[attr_start:attr_stop]
        sub_attr_normed[sub_mod_name] = this_ensem_attr_normed        
        sub_attr_total[sub_mod_name] = this_ensem_attr.sum(dim=1)
        attr_start = attr_stop
        max_hidden = max(max_hidden, hidden_size[i])
        min_hidden = min(min_hidden, hidden_size[i])
        max_attr = max(max_attr, this_ensem_attr_normed.max())
        min_attr = min(min_attr, this_ensem_attr_normed.min())
    # Pad and stack for plotting
    sub_attr_stack = []
    sub_tot_attr_stack = []
    sub_names = []
    pad_value = min_attr - ((max_attr - min_attr) * 0.1) 
    for name in sub_attr_normed.keys():
        deficit = max_hidden - sub_attr_normed[name].numel()
        pads = (ceil(deficit/2), floor(deficit/2))
        padded = F.pad(sub_attr_normed[name], pads, mode='constant', value=pad_value).unsqueeze(0)
        sub_attr_stack.append(padded.numpy())
        sub_names.append(name)
        sub_tot_attr_stack.append(sub_attr_total[name].unsqueeze(0).numpy())
        ic(sub_attr_total[name].numpy())
    sub_attrs_plot = np.concatenate(sub_attr_stack)
    sub_tot_attrs_plot = np.concatenate(sub_tot_attr_stack).T
    # Visualize
    fig = plt.figure(figsize=(20,6))
    gs = fig.add_gridspec(1, 2, left=0.05, right=0.95, bottom=0.1, top=0.75, wspace=0.1)
    ax = fig.add_subplot(gs[0, 0])
    color_res = 256
    viridis = cm.get_cmap('viridis', color_res)
    custom_colors = viridis(np.linspace(0,1,color_res))
    invalid_color = np.array([150/color_res, 75/color_res, 150/color_res, 1])
    invalid_bound = floor(0.05 * color_res)
    custom_colors[:invalid_bound] = invalid_color
    custom_cm = ListedColormap(custom_colors)
    pc = ax.pcolor(sub_attrs_plot, cmap=custom_cm, rasterized=True)
    #img = ax.imshow(sub_attrs_plot, cmap=custom_cm)
    ax.set_aspect(50.0)
    ax.xaxis.set_ticks_position("none")
    ax.set_xticklabels([])
    ax.set_yticks(np.arange(len(sub_names))+0.5, labels=sub_names)
    ax.yaxis.tick_right()
    ax.set_title("Avg. Ensemble Layer Attribution by Sub-Model")
    cb = fig.colorbar(pc, ax=ax, location='left')
    #plt.grid(axis='y')
    #fig.tight_layout()
    
    ax2 = fig.add_subplot(gs[0,1])
    
    ax2.boxplot(sub_tot_attrs_plot, vert=False)

    ax2.set_title("Total Attribution per Sub-Model")
    
    fig.savefig(os.path.join(save_dir, save_name))


def visualize_attributions(attrs, inputs, y_hat, y, tokenizer, sub_models, model_name,
    save_dir="data/08_reporting", save_name="tmp.png", ensemble=False):
    if ensemble:
        visualize_model_attributions(attrs, inputs, y_hat, y, sub_models, model_name, 
            save_dir, save_name)
    else:
        visualize_input_attributions(attrs, inputs, y_hat, y, tokenizer, model_name, 
            save_dir, save_name)
   

@click.command
@click.option('--model_name', default='visual-bert')
@click.option('--ckpt_dir', default='data/06_models/visual_bert')
@click.option('--no_save', is_flag=True, help='Flag for disabling vis. saving (e.g. for testing)')
@click.option('--save_dir', default='data/08_reporting')
@click.option('--save_name', default='tmp.png')
@click.option('--ensemble', is_flag=True, help='Perform model attribution for an ensemble')
def interp(model_name:str, ckpt_dir:str, no_save:bool, save_dir:str, save_name:str,
    ensemble:bool):
    
    ## DataLoader
    B = 50 if ensemble else 1
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
        attrs = get_model_attributions(interp_model, data_sample)
    else:
        attrs = get_input_attributions(interp_model, data_sample)
    
    ## Visualize attributions
    if(~no_save):
        inputs = {
            'img':data_sample["image"],
            'txt':data_sample["text"]
        }
        y_hat = interp_model.inner_model(data_sample)
        y = data_sample["label"] 
        visualize_attributions(attrs, inputs, y_hat, y, interp_model.tokenizer,
            interp_model.sub_models, model_name, save_dir, save_name, ensemble)


if __name__ == '__main__':
    interp()