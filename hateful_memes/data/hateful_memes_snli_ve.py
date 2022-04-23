import base64
from math import floor
from pathlib import Path
import pandas as pd
import torch
import pytorch_lightning as pl
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from io import BytesIO
import os
import multiprocessing as mp


class XformBase64():
    def __init__(self, print_freq=0):
        self.print_freq=print_freq
        self.call_num = 0

    def __call__(self, img_str:str):
        self.call_num = self.call_num + 1
        if (self.print_freq != 0) and (self.call_num % self.print_freq == 0):
            print("Transforming %ith image" % self.call_num)
        img = Image.open(img_str)
        intermed_buf = BytesIO()
        img.save(intermed_buf, format="PNG")
        img_b64_str = base64.b64encode(intermed_buf.getvalue())
        return img_b64_str.decode('utf-8')


class XformAbsPath():
    def __init__(self, root_dir:str):
        self.root = root_dir
    def __call__(self, img_str:str):
        new_str = os.path.join(self.root, img_str)
        return new_str


def mp_xform_b64(xform_tuple):
    df = xform_tuple[0]
    xform = xform_tuple[1]
    return df.transform(xform)


def convert_to_snli_ve(set_name:str, json_name:str, hateful_memes_dir=None, tsv_save_name=None):
    """
    Original format:
    "id", image id / "img", path to img / "label", binary class / "text", text caption
    
    Target format:
    "unique_id", image id + hash? / "image_id", image id / "image", image as base64 string / "hypothesis", text hypothesis / "caption", text caption / "label", label
    """
    # Grab original dataset
    load_dir = Path(hateful_memes_dir) if hateful_memes_dir is not None else Path("data/01_raw/hateful_memes")
    raw_memes= pd.read_json(load_dir/json_name, lines=True)

    # copy "id" column, insert at position 1 as "image_id"
    # rename column 0 as unique_id
    uid_col = raw_memes["id"]
    iid_col = raw_memes["id"]
    snli_memes = pd.concat({"unique_id":uid_col, "image_id":iid_col}, axis=1)
    
    # transform "img" from image file path to image base64 string, rename as "image"
    xform_to_abs = XformAbsPath(load_dir)
    xform_to_base64 = XformBase64(200)
    snli_memes["image"] = raw_memes["img"].transform(xform_to_abs)
    # This next bit is pretty slow, so leverage multiple workers
    num_procs = mp.cpu_count() - 1
    if num_procs > 1:
        worker_pool = mp.Pool(num_procs)
        num_imgs = snli_memes.shape[0]
        imgs_per_proc = floor(num_imgs/(num_procs - 1))  # last proc gets remainder, which may be more/less than imgs_per_proc
        num_imgs_split = imgs_per_proc * (num_procs - 1)
        stack_of_splits = [(snli_memes["image"][i - imgs_per_proc : i], XformBase64(200))
            for i in range(imgs_per_proc, num_imgs_split+1, imgs_per_proc)]
        last_split = (snli_memes["image"][num_imgs_split:num_imgs], XformBase64(200))
        stack_of_splits.append(last_split)
        xform_stack = worker_pool.map(mp_xform_b64, stack_of_splits)
        snli_memes["image"] = pd.concat(xform_stack)
    else:    
        snli_memes["image"] = snli_memes["image"].transform(xform_to_base64)

    # insert "hypothesis" column at position 3
    hypothesis = "I am hateful."
    snli_memes["hypothesis"] = pd.DataFrame({"hypothesis":[hypothesis,]*snli_memes.shape[0]})

    # Add caption column
    snli_memes["caption"] = raw_memes["text"]
    
    # Move "label" to end, transform from binary to "entailment"/"contradiction" 
    def xform_to_entailment(bin_class:int):
        label = ""
        if bin_class == 1:
            label = "entailment"
        elif bin_class == 0:
            label = "contradiction"
        return label
    
    snli_memes["label"] = raw_memes["label"].transform(xform_to_entailment)

    # Save to TSV for OFA
    save_name = Path(tsv_save_name) if tsv_save_name is not None else Path("data/02_intermediate/hateful_memes_%s_snli_ve.tsv" % (set_name))
    with open(save_name, 'w') as tsv:
        snli_memes.to_csv(tsv, sep="\t", index=False, header=False)


def main():
    # training set
    convert_to_snli_ve(set_name="train", json_name="train.jsonl")
    # validation set
    convert_to_snli_ve(set_name="valid", json_name="dev_seen.jsonl")


if __name__ == "__main__":
    main()