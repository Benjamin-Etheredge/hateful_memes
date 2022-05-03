from pathlib import Path
import os
import glob
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
from torchvision.transforms import PILToTensor, ToPILImage


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
            self.image_embed_layer = self.inner_model.backbone[1]
            self.attr_image_input = True
            self.text_embed_layer = self.inner_model.backbone[0].embeddings.word_embeddings
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
        elif model_name == 'distilbert':
            self.inner_model = AutoTextModule.load_from_checkpoint(checkpoint_path=ckpt_path)
            self.text_embed_layer = self.inner_model.model.embeddings.word_embeddings
            self.attr_text_input = False
            self.tokenizer = self.inner_model.tokenizer        
        elif model_name == 'visual-bert-with-od':
            self.inner_model = VisualBertWithODModule.load_from_checkpoint(checkpoint_path=ckpt_path)
            self.image_embed_layer = self.inner_model.backbone[1]
            self.attr_image_input = True
            self.text_embed_layer = self.inner_model.backbone[0].embeddings.word_embeddings
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
 