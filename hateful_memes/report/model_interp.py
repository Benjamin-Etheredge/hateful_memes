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
            self.inner_model.eval()
            self.image_embed_layer = self.inner_model.resnet
            self.text_embed_layer = self.inner_model.visual_bert.embeddings.word_embeddings
            self.tokenizer = self.inner_model.tokenizer
        elif model_name == 'beit':
            self.inner_model = BaseITModule.load_from_checkpoint(checkpoint_path=ckpt_path)
            self.inner_model.eval()
            self.image_embed_layer = self.inner_model.feature_extractor
        elif model_name == 'electra':
            self.inner_model == AutoTextModule.load_from_checkpoint(checkpoint_path=ckpt_path)
            self.inner_model.eval()
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
    

def wrapper_forward(image, input_ids, tokenizer, pl_module):
    #ic(image)
    #ic(input_ids)
    #ic(tokenizer_inner_model)
    #tokenizer=tokenizer_pl_module[0]
    #pl_module=tokenizer_pl_module[1]
    # reassemble dict
    text_orig = [tokenizer.decode(input_id, skip_special_tokens=True)
        for input_id in input_ids.tolist()]
    #ic(text_orig)
    batch = {
        'image': image,
        'text': text_orig
    }
    #ic(batch)
    return pl_module(batch)


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


def visualize_attributions(attrs, inputs, y_hat, y_label, tokenizer,
    save_results=False, save_dir="data/08_reporting", save_name="tmp.png"):
    if 'img' in attrs.keys():
        img_attr = attrs['img']
        img_in = inputs['img']
        img_attr_vis = img_attr.squeeze(dim=0).permute(1, 2, 0).numpy()
        img_in_vis = img_in.squeeze(dim=0).permute(1, 2, 0).numpy()
        fig, ax = visualization.visualize_image_attr(img_attr_vis, original_image=img_in_vis,
            method="blended_heat_map", sign="all", show_colorbar=True)
        if(save_results):
           fig.savefig(os.path.join(save_dir, save_name))

    if 'txt' in attrs.keys():
        # Look at https://captum.ai/tutorials/Multimodal_VQA_Interpret
        # something = visualization.VisualizationDataRecord(...)
        txt_attr = attrs['txt'].squeeze(0)  # Should now be (max_length x embed_dim)
        txt_in = inputs['txt']
        token_dict = tokenizer(
            txt_in, 
            return_tensors="pt", 
            padding='max_length', 
            truncation=True, 
            max_length=txt_attr.shape[0],
            add_special_tokens=True, return_special_tokens_mask=True)
        
        ic(txt_in)

        token_mask = token_dict['special_tokens_mask'] < 1
        txt_attr = txt_attr.sum(-1)
        txt_attr = torch.masked_select(txt_attr, token_mask.squeeze(0))
        txt_attr_norm = txt_attr.norm()
        txt_attr_normed = txt_attr/txt_attr_norm
        y_hat_label = "Hateful" if y_hat>0.5 else "Not Hateful" 


        fig, ax = plt.subplots()
        txt_img = ax.imshow(txt_attr_normed.unsqueeze(-1).numpy())
        ax.set_yticks(np.arange(txt_attr_normed.numel()), labels=txt_in[0].split())
        for idx, attr_i in enumerate(txt_attr_normed.tolist()):
            ax.text(0, idx, '%.3g' % (attr_i,), ha="center", va="center", color="w")
        ax.figure.colorbar(txt_img, ax=ax)
        ax.axes.xaxis.set_ticklabels([])
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "tmp_txt.png"))
        """
        junk = Image.new("RGB", (1,1))
        draw = ImageDraw.Draw(junk)
        width, height = draw.textsize(text=txt_in)
        mline_text = txt_in[0]
        width, height = draw.multiline_textsize(text=mline_text)
        spacing = 20
        text_vis = Image.new("RGB", ((2*spacing) + width, (2*spacing) + height), (255, 255, 255))
        text_vis_font = ImageFont.load_default()
        draw_ctx = ImageDraw.Draw(text_vis)
        pos_color = (0, 255, 0)
        neg_color = (255, 0, 0)
        paste_loc = (spacing, spacing)
        for idx, word in enumerate(txt_in[0].split()):
            word_w, word_h = draw_ctx.textsize(word)
            word_color = (255,255,255)
            if txt_attr_normed[idx] > 0:
                word_alpha = round(txt_attr_normed[idx].item() * 255.)
                word_color = pos_color + (word_alpha,)
            else:
                word_alpha = round(abs(txt_attr_normed[idx].item() * 255.))
                word_color = neg_color + (word_alpha,)
            word_img = Image.new("RGBA", (word_w, word_h), word_color)
            text_vis.paste(word_img, paste_loc)
            #draw_ctx.text(paste_loc, word + "\n", font=text_vis_font)
            paste_loc = (paste_loc[0], paste_loc[1] + word_h) 
        draw_ctx.multiline_text(text=mline_text, xy=(spacing,spacing), spacing=4, font=text_vis_font)
        #vis_rec = visualization.VisualizationDataRecord(txt_attr/txt_attr_norm, y_hat, y_hat_label, y_label, y_hat_label,
        #    txt_attr.sum(), txt_in[0].split(), 0.0)
        #html = visualization.visualize_text([vis_rec], legend=True)

        if(save_results):
            text_vis.save(os.path.join(save_dir, "tmp_txt.png"))
        #ic(txt_attr.shape)
        #ic(txt_attr)
        """

    total_attr = img_attr.sum().item() + txt_attr.sum().item()
    ic("ATTR:", total_attr)
    ic("PRED:", y_hat)
       

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
    inputs = {
        'img':data_sample["image"],
        'txt':data_sample["text"]
    }
    y_hat = interp_model.inner_model(data_sample).item()
    y_label = "Hateful" if data_sample["label"]>0 else "Not Hateful" 
    visualize_attributions(attrs, inputs, y_hat, y_label, interp_model.tokenizer,
        ~no_save, save_dir, save_name)


if __name__ == '__main__':
    interp()