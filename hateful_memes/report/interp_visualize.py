import matplotlib.pyplot as plt
import numpy as np
from math import floor, ceil
import torch
import sys
import os
from icecream import ic
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
import click
import json
from pathlib import Path

from hateful_memes.report.model_wrappers import InterpModel


def rgb_to_grey(rgb):
    grey_coeffs = np.array([[[0.2989, 0.5870, 0.1140]]])
    grey = (rgb * grey_coeffs).sum(axis=2)
    return grey


def visualize_input_attributions(attrs, inputs, y_hat, y, tokenizer, model_name,
    save_name="tmp.png"):
    
    # Prediction string
    y_hat = y_hat.item()
    y_hat_prob = F.sigmoid(y_hat)
    pred = 1 if y_hat_prob>=0.5 else 0
    y_hat_label = "Hateful" if pred==1 else "Not Hateful" 
    y_label = "Hateful" if y==1 else "Not Hateful" 
    pn_str = "positive"
    tf_str = "false"
    if y==pred:
        tf_str = "true"
    if pred==0:
        pn_str = "negative"
    pred_str = "Model predicted: \"%s\", w/ prob. %.3g\n(%s %s)" % (y_hat_label, y_hat_prob, tf_str, pn_str)

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
    fig.savefig(save_name)
       

def visualize_model_attributions(attrs, inputs, y_hat, y, sub_models, model_name, 
    save_name="tmp.png"):
    ensem_attr = attrs['models']
    ensem_attr_mean = ensem_attr.mean(dim=0)
    ensem_attr_normed = ensem_attr_mean/ensem_attr_mean.norm()
    # Masks for results
    y_hat_prob = torch.sigmoid(y_hat)
    pred = y_hat_prob>=0.5
    tp_idx = np.nonzero(((pred == y) & (pred == 1)).numpy())
    tn_idx = np.nonzero(((pred == y) & (pred == 0)).numpy())
    fp_idx = np.nonzero(((pred != y) & (pred == 1)).numpy())
    fn_idx = np.nonzero(((pred != y) & (pred == 0)).numpy())
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
        sub_mod_name = sub.plot_name + ("(%i)"%(i+1,))
        attr_stop = attr_start + hidden_size[i]
        this_ensem_attr = ensem_attr[:, attr_start:attr_stop]
        sub_attr[sub_mod_name] = this_ensem_attr        
        this_ensem_attr_normed = ensem_attr_normed[attr_start:attr_stop]
        sub_attr_normed[sub_mod_name] = this_ensem_attr_normed        
        # Here, we define a sub-model's total attribution as
        # the sum of the ensemble-layer attributions that
        # belong to this sub-model
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
    sub_attrs_plot = np.concatenate(sub_attr_stack)
    sub_tot_attrs_plot = np.concatenate(sub_tot_attr_stack).T
    
    # Visualize
    #fig = plt.figure(figsize=(20,6))
    #gs = fig.add_gr(idspec(1, 2, left=0.05, right=0.95, bottom=0.1, top=0.75, wspace=0.3)
    # Average ensemble-layer contributions per sub-model
    fig0 = plt.figure(figsize=(11,6))
    gs0 = fig0.add_gridspec(1, 1, left=0.25, right=0.95)
    ax0 = fig0.add_subplot(gs0[0,0])
    color_res = 256
    viridis = cm.get_cmap('viridis', color_res)
    custom_colors = viridis(np.linspace(0,1,color_res))
    invalid_color = np.array([150/color_res, 75/color_res, 150/color_res, 1])
    invalid_bound = floor(0.05 * color_res)
    custom_colors[:invalid_bound] = invalid_color
    custom_cm = ListedColormap(custom_colors)
    pc = ax0.pcolor(sub_attrs_plot, cmap=custom_cm, rasterized=True)
    ax0.set_aspect(50.0)
    #ax0.xaxis.set_ticks_position("none")
    ax0.set_xticks(np.arange(0, max_hidden, 64), labels=[str(_x) for _x in range(0, max_hidden, 64)])
    ax0.set_xlabel("Ensemble layer section size (neurons)")
    ax0.set_yticks(np.arange(len(sub_names))+0.5, labels=sub_names)
    ax0.set_ylabel("Sub-model name")
    #ax0.yaxis.tick_right()
    ax0.set_title("Avg. Ensemble Layer Attribution by Sub-Model")
    cb = fig0.colorbar(pc, ax=ax0, location='right')
    save0 = save_name + "_0.png"
    fig0.savefig(save0)

    # Absolute total attribution per sub-model over N runs
    fig1 = plt.figure(figsize=(11, 6))
    gs1 = fig1.add_gridspec(1, 1, left=0.25, right=0.95)
    ax1 = fig1.add_subplot(gs1[0,0])
    #ax1.set_xticks(np.arange(0, max_hidden, 64), labels=[str(_x) for _x in range(0, max_hidden, 64)])
    ax1.boxplot(sub_tot_attrs_plot, vert=False)
    ax1.set_yticklabels(sub_names)
    ax1.set_ylabel("Sub-model name")
    ax1.set_xlabel("Total attribution to logit")
    ax1.set_title("Total Attribution per Sub-Model")
    save1 = save_name + "_1.png"    
    fig1.savefig(save1)

    # Absolute total attribution per sub-model by outcome
    tp_attr = sub_tot_attrs_plot[tp_idx]
    tn_attr = sub_tot_attrs_plot[tn_idx]
    fp_attr = sub_tot_attrs_plot[fp_idx]
    fn_attr = sub_tot_attrs_plot[fn_idx]
    
    fig2 = plt.figure(figsize=(17,8))
    gs2 = fig2.add_gridspec(2, 2, left=0.1, right=0.95, wspace=0.35)
    ax2_tp = fig2.add_subplot(gs2[0,0])
    ax2_tp.boxplot(tp_attr, vert=False) 
    ax2_tp.set_yticklabels(sub_names)
    ax2_tp.set_title("True Positives")
    ax2_tn = fig2.add_subplot(gs2[0,1]) 
    ax2_tn.boxplot(tn_attr, vert=False) 
    ax2_tn.set_yticklabels(sub_names)
    ax2_tn.set_title("True Negatives")
    ax2_fp = fig2.add_subplot(gs2[1,0]) 
    ax2_fp.boxplot(fp_attr, vert=False) 
    ax2_fp.set_yticklabels(sub_names)
    ax2_fp.set_title("False Positives")
    ax2_fn = fig2.add_subplot(gs2[1,1]) 
    ax2_fn.boxplot(fn_attr, vert=False) 
    ax2_fn.set_yticklabels(sub_names)
    ax2_fn.set_title("False Negatives")
    fig2.suptitle("Total Logit Attribution Per Sub-Model")
    save2 = save_name + "_2.png"
    fig2.savefig(save2)


def visualize_attributions(attrs, inputs, y_hat, y, tokenizer, sub_models, model_name,
    save_name="tmp.png", ensemble=False):
    if ensemble:
        visualize_model_attributions(attrs, inputs, y_hat, y, sub_models, model_name, 
            save_name)
    else:
        visualize_input_attributions(attrs, inputs, y_hat, y, tokenizer, model_name, 
            save_name)


@click.command
@click.option('--attr_file', default='data/08_reporting/tmp.pt', help='PT file in which attribution scores are stored')
@click.option('--save_dir', default='data/08_reporting')
@click.option('--save_prefix', default='tmp')
@click.option('--ensemble', is_flag=True, help='Visualize model attribution for an ensemble')
def visualize(attr_file:str, save_dir:str, save_prefix:str, ensemble:bool):
    ## Retrive data sample
    data_sample = torch.load(attr_file)
    model_name = data_sample['model_name']
    ckpt_dir = data_sample['ckpt_dir']
   
    ## Model to interpret
    interp_model = InterpModel(model_name, ckpt_dir) 

    ## Gather inputs 
    inputs = {
        'img':data_sample["image"],
        'txt':data_sample["text"]
    }
    y_hat = interp_model.inner_model(data_sample)
    y = data_sample["label"] 

    ## Set up outdir
    if ~Path(save_dir).exists():
        os.mkdir(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)

    ## Visualize
    visualize_attributions(data_sample, inputs, y_hat, y, interp_model.tokenizer,
        interp_model.sub_models, model_name, save_prefix, ensemble)


if __name__ == '__main__':
    visualize()