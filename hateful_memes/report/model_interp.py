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


class InterpModel():
    def __init__(self, model_name:str, ckpt_dir:str):
        ## True model (Just Visual BERT for now)
        assert(Path(ckpt_dir).exists())
        ckpt_search = os.path.join(ckpt_dir, "*.ckpt")
        ckpt_path = glob.glob(ckpt_search)[0]  # Grab the most recent checkpoint?
        self.inner_model = None
        self.image_embed_layer = None
        self.text_embed_layer = None
        self.tokenizer = None
        if model_name == '':
            raise ValueError
        elif model_name == 'visual-bert':
            self.inner_model = VisualBertModule.load_from_checkpoint(checkpoint_path=ckpt_path)
            self.image_embed_layer = self.inner_model.resnet
            self.text_embed_layer = self.inner_model.visual_bert.embeddings.word_embeddings
            self.tokenizer = self.inner_model.tokenizer
        elif model_name == 'beit':
            self.inner_model = BaseITModule.load_from_checkpoint(checkpoint_path=ckpt_path)
            in_wrap = ModelInputWrapper(self.inner_model.model.beit)
            self.inner_model.model.beit = in_wrap
            #self.image_embed_layer = self.inner_model.model.beit.embeddings.patch_embeddings
            self.image_embed_layer = in_wrap.input_maps["pixel_values"]
        elif model_name == 'electra':
            self.inner_model = AutoTextModule.load_from_checkpoint(checkpoint_path=ckpt_path)
            self.text_embed_layer = self.inner_model.model.embeddings.word_embeddings
            self.tokenizer = self.inner_model.tokenizer
            
    # Used as wrapper for model forward()
    def __call__(self, image, input_ids, tokenizer):
        # reassemble dict
        text_orig = [tokenizer.decode(input_id, skip_special_tokens=True)
            for input_id in input_ids.tolist()]
        #ic(text_orig)
        batch = {
            'image': image,
            'text': text_orig
        }
        #ic(batch)
        return self.inner_model(batch)
    

def get_attributions(interp_model:InterpModel, data_sample):
    ## Calculate feature attribution
    # Features
    image = data_sample['image']
    text = data_sample['text']
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    token_dict = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    token_ids = token_dict['input_ids']
    image_text = (image, token_ids)
    # Feature masks (?)
    image_mask = torch.tensor(felzenszwalb(image.squeeze(dim=0).numpy(), channel_axis=0,
        scale=0.25, min_size=5))
    text_mask = torch.arange(token_ids.numel()) + image_mask.max()
    # Feature baselines
    image_baselines = torch.zeros_like(image)
    token_ref= TokenReferenceBase(reference_token_idx=tokenizer.convert_tokens_to_ids("[PAD]"))
    text_baselines = token_ref.generate_reference(token_ids.numel(), device='cpu').unsqueeze(0)
    # Layer Integrated Gradients
    attrs = {}
    interp_model.inner_model.eval()
    if interp_model.image_embed_layer is not None:
        lig_image = LayerIntegratedGradients(interp_model, interp_model.image_embed_layer)
        img_attr = lig_image.attribute(inputs=image_text,
            baselines=(image_baselines, text_baselines),
            additional_forward_args=tokenizer,
            attribute_to_layer_input=True)
        attrs['img'] = img_attr
    if interp_model.text_embed_layer is not None:
        lig_txt = LayerIntegratedGradients(interp_model, interp_model.text_embed_layer)
        txt_attr = lig_txt.attribute(inputs=image_text,
            baselines=(image_baselines, text_baselines),
            additional_forward_args=tokenizer)
        attrs['txt'] = txt_attr
    
    return attrs


def visualize_attributions(attrs, inputs, y_hat, y, tokenizer, model_name,
    save_dir="data/08_reporting", save_name="tmp.png"):
    
    # Prediction string
    pred = 1 if y_hat>0.5 else 0
    y_hat_label = "Hateful" if pred==1 else "Not Hateful" 
    y_label = "Hateful" if y==1 else "Not Hateful" 
    pn_str = "positive"
    tf_str = "false"
    if y==pred:
        tf_str = "true"
    if pred==0:
        pn_str = "negative"
    pred_str = "Model predicted: \"%s\", probability %.3g\n(%s %s)" % (y_hat_label, y_hat, tf_str, pn_str)

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
    fig = plt.figure(figsize=(12, 5), constrained_layout=False)
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
    txt_attr = None
    # Visualize image feature attributions
    if 'img' in attrs.keys():
        img_attr = attrs['img']
        img_attr_vis = img_attr.squeeze(dim=0).permute(1, 2, 0).numpy()
        axi = fig.add_subplot(gs[0, current_sub])
        current_sub += 1
        img_attr_comb = np.sum(img_attr_vis, axis=-1)
        img_attr_normed = img_attr_comb/np.linalg.norm(img_attr_comb)
        imi = axi.imshow(img_attr_normed)
        axi.xaxis.set_ticks_position("none")
        axi.yaxis.set_ticks_position("none")
        axi.set_xticklabels([])
        axi.set_yticklabels([])        
        axi.set_title("Norm. Visual Attr.")
        fig.colorbar(imi, ax=axi)
    
    # Visualize text feature attributions
    if 'txt' in attrs.keys():
        txt_attr = attrs['txt'].squeeze(0)  # Should now be (max_length x embed_dim)
        txt_in = inputs['txt']
        token_dict = tokenizer(
            txt_in, 
            return_tensors="pt", 
            padding='max_length', 
            truncation=True, 
            max_length=txt_attr.shape[0],
            add_special_tokens=True, return_special_tokens_mask=True)
        
        token_mask = token_dict['special_tokens_mask'] < 1
        txt_attr = txt_attr.sum(-1)
        txt_attr = torch.masked_select(txt_attr, token_mask.squeeze(0))
        txt_attr_norm = txt_attr.norm()
        txt_attr_normed = txt_attr/txt_attr_norm
        
        axt = fig.add_subplot(gs[0, current_sub+1])
        current_sub += 3
        imt = axt.imshow(txt_attr_normed.unsqueeze(-1).numpy())
        axt.set_yticks(np.arange(txt_attr_normed.numel()), labels=txt_in[0].split())
        for idx, attr_i in enumerate(txt_attr_normed.tolist()):
            axt.text(0, idx, '%.2g' % (attr_i,), ha="center", va="center", color="w")
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
       

@click.command
@click.option('--model_name', default='visual-bert')
@click.option('--ckpt_dir', default='data/06_models/visual_bert')
@click.option('--no_save', is_flag=True, help='Flag for disabling vis. saving (e.g. for testing)')
@click.option('--save_dir', default='data/08_reporting')
@click.option('--save_name', default='tmp.png')
def interp(model_name:str, ckpt_dir:str, no_save:bool, save_dir:str, save_name:str):
    ## DataLoader
    datamodule = MaeMaeDataModule(batch_size=1) # Attributors want one sample at a time?
    datamodule.prepare_data()
    datamodule.setup("fit")
    dataloader = datamodule.train_dataloader()    
    data_sample = next(iter(dataloader))
    ## Model to interpret
    interp_model = InterpModel(model_name, ckpt_dir)
    ## Get attributions
    attrs = get_attributions(interp_model, data_sample)
    ## Visualize attributions
    if(~no_save):
        inputs = {
            'img':data_sample["image"],
            'txt':data_sample["text"]
        }
        y_hat = interp_model.inner_model(data_sample).item()
        y = data_sample["label"] 
        visualize_attributions(attrs, inputs, y_hat, y, interp_model.tokenizer,
            model_name, save_dir, save_name)


if __name__ == '__main__':
    interp()