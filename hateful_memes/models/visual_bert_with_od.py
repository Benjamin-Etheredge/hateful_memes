import click
from icecream import ic

import torch
from torch.nn import functional as F
from torch import nn
import torchvision.models as models
import torchvision.transforms 

from transformers import BertTokenizer, VisualBertModel
# from transformers import DetrFeatureExtractor, DetrForObjectDetection, AutoConfig
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures.image_list import ImageList
from detectron2.data import transforms as T
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputs
from detectron2.structures.boxes import Boxes
from detectron2.layers import nms
from detectron2 import model_zoo
from detectron2.config import get_cfg

import cv2

import pytorch_lightning as pl

from hateful_memes.models.base import BaseMaeMaeModel, base_train

class Detectron2Module():
    def __init__(self, batch_size, num_queries):
        self.cfg_path = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
        self.cfg = self.load_config_and_model_weights(self.cfg_path)
        self.model = self.get_model(self.cfg)
        self.MIN_BOXES=10
        self.MAX_BOXES=num_queries
        self.batch_size = batch_size

    def load_config_and_model_weights(self, cfg_path):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(cfg_path))

        # ROI HEADS SCORE THRESHOLD
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

        # Comment the next line if you're using 'cuda'
        cfg['MODEL']['DEVICE']='cpu'

        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_path)

        return cfg
    
    def get_model(self, cfg):
        # build model
        model = build_model(cfg)

        # load weights
        checkpointer = DetectionCheckpointer(model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        # eval mode
        model.eval()
        return model

    def prepare_image_inputs(self, cfg, img_list):
        # Resizing the image according to the configuration
        transform_gen = T.ResizeShortestEdge(
                    [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
                )
        img_list = [transform_gen.get_transform(img).apply_image(img) for img in img_list]

        # Convert to C,H,W format
        convert_to_tensor = lambda x: torch.Tensor(x.astype("float32").transpose(2, 0, 1))

        batched_inputs = [{"image":convert_to_tensor(img), "height": img.shape[0], "width": img.shape[1]} for img in img_list]

        # Normalizing the image
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(num_channels, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).view(num_channels, 1, 1)
        normalizer = lambda x: (x - pixel_mean) / pixel_std
        images = [normalizer(x["image"]) for x in batched_inputs]

        # Convert to ImageList
        images =  ImageList.from_tensors(images, self.model.backbone.size_divisibility)
        
        return images, batched_inputs

    def get_features(self, model, images):
        features = model.backbone(images.tensor)
        return features

    def get_proposals(self, model, images, features):
        proposals, _ = model.proposal_generator(images, features)
        return proposals
    
    def get_box_features(self, model, features, proposals):
        features_list = [features[f] for f in ['p2', 'p3', 'p4', 'p5']]
        box_features = model.roi_heads.box_pooler(features_list, [x.proposal_boxes for x in proposals])
        box_features = model.roi_heads.box_head.flatten(box_features)
        box_features = model.roi_heads.box_head.fc1(box_features)
        box_features = model.roi_heads.box_head.fc_relu1(box_features)
        box_features = model.roi_heads.box_head.fc2(box_features)

        box_features = box_features.reshape(self.batch_size, 1000, 1024) # depends on your config and batch size
        return box_features, features_list
    

    def get_prediction_logits(self, model, features_list, proposals):
        cls_features = model.roi_heads.box_pooler(features_list, [x.proposal_boxes for x in proposals])
        cls_features = model.roi_heads.box_head(cls_features)
        pred_class_logits, pred_proposal_deltas = model.roi_heads.box_predictor(cls_features)
        return pred_class_logits, pred_proposal_deltas

    
    def get_box_scores(self, cfg, pred_class_logits, pred_proposal_deltas):
        box2box_transform = Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)
        smooth_l1_beta = cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA

        outputs = FastRCNNOutputs(
            box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            smooth_l1_beta,
        )

        boxes = outputs.predict_boxes()
        scores = outputs.predict_probs()
        image_shapes = outputs.image_shapes

        return boxes, scores, image_shapes

    def get_output_boxes(self, boxes, batched_inputs, image_size):
        proposal_boxes = boxes.reshape(-1, 4).clone()
        scale_x, scale_y = (batched_inputs["width"] / image_size[1], batched_inputs["height"] / image_size[0])
        output_boxes = Boxes(proposal_boxes)

        output_boxes.scale(scale_x, scale_y)
        output_boxes.clip(image_size)

        return output_boxes

    def select_boxes(self, cfg, output_boxes, scores):
        test_score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
        test_nms_thresh = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
        cls_prob = scores.detach()
        cls_boxes = output_boxes.tensor.detach().reshape(1000,80,4)
        max_conf = torch.zeros((cls_boxes.shape[0]))
        for cls_ind in range(0, cls_prob.shape[1]-1):
            cls_scores = cls_prob[:, cls_ind+1]
            det_boxes = cls_boxes[:,cls_ind,:]
            keep = np.array(nms(det_boxes, cls_scores, test_nms_thresh))
            max_conf[keep] = torch.where(cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep])
        keep_boxes = torch.where(max_conf >= test_score_thresh)[0]
        return keep_boxes, max_conf


    def filter_boxes(self, keep_boxes, max_conf, min_boxes, max_boxes):
        if len(keep_boxes) < min_boxes:
            keep_boxes = np.argsort(max_conf).numpy()[::-1][:min_boxes]
        elif len(keep_boxes) > max_boxes:
            keep_boxes = np.argsort(max_conf).numpy()[::-1][:max_boxes]
        return keep_boxes

    def get_visual_embeds(self, img_list):
        images, batched_inputs = self.prepare_image_inputs(self.cfg, img_list)
        features = self.get_features(self.model, images)
        proposals = self.get_proposals(self.model, images, features) 
        box_features, features_list = self.get_box_features(self.model, features, proposals)
        pred_class_logits, pred_proposal_deltas = self.get_prediction_logits(self.model, features_list, proposals)
        boxes, scores, image_shapes = self.get_box_scores(self.cfg, pred_class_logits, pred_proposal_deltas)
        output_boxes = [self.get_output_boxes(boxes[i], batched_inputs[i], proposals[i].image_size) for i in range(len(proposals))]
        temp = [self.select_boxes(cfg, output_boxes[i], scores[i]) for i in range(len(scores))]
        keep_boxes, max_conf = [],[]
        for keep_box, mx_conf in temp:
            keep_boxes.append(keep_box)
            max_conf.append(mx_conf)
        keep_boxes = [self.filter_boxes(keep_box, mx_conf, self.MIN_BOXES, self.MAX_BOXES) for keep_box, mx_conf in zip(keep_boxes, max_conf)]
        visual_embeds = [self.get_visual_embeds(box_feature, keep_box) for box_feature, keep_box in zip(box_features, keep_boxes)]
        return torch.stack(visual_embeds)
class VisualBertWithODModule(BaseMaeMaeModel):
    """ Visual Bert Model """

    def __init__(
        self,
        lr=0.003,
        max_length=512,
        include_top=True,
        dropout_rate=0.0,
        dense_dim=256,
        num_queries=50,
        batch_size=32,
        freeze=False,
    ):
        """ Visual Bert Model """
        super().__init__()
        # self.hparams = hparams
        self.visual_bert = VisualBertModel.from_pretrained("uclanlp/visualbert-vqa-coco-pre")
        if freeze:
            for param in self.visual_bert.parameters():
                param.requires_grad = False
            self.visual_bert.eval()
        # ic(self.visual_bert)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        ############################################
        # Obj Detection Start
        ############################################
        self.od_config = AutoConfig.from_pretrained('facebook/detr-resnet-50', num_queries=num_queries)
        self.od_feature_extractor = DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-50')
        self.od_model = DetrForObjectDetection(self.od_config)
        # ic(self.od_config.num_queries)
        # self.od_fc = nn.Linear(self.od_config.num_queries * 256, 2048)
        # self.od_fc = nn.Sequential(
        #     nn.Linear(256, 2048),
        #     # nn.BatchNorm2d(2048),
        #     nn.Tanh(),
        #     # nn.AdaptiveAvgPool1d(1)
        # )
        # self.od_fc = nn.Linear(256, 768)

        self.detr2 = Detectron2Module(batch_size=batch_size, num_queries=num_queries)
         

        if freeze:
            for param in self.od_model.parameters():
                param.requires_grad = False
            self.od_model.eval()
        ############################################
        # Obj Detection End
        ############################################

        # TODO linear vs embedding for dim changing
        # TODO auto size
        self.fc1 = nn.Linear(768, dense_dim)
        self.fc2 = nn.Linear(dense_dim, dense_dim)
        self.fc3 = nn.Linear(dense_dim, 1)
        # TODO config modification

        self.lr = lr
        self.max_length = max_length
        self.include_top = include_top
        self.dropout_rate = dropout_rate
        self.dense_dim = dense_dim
        self.to_freeze = freeze
        self.visual_bert_config = self.visual_bert.config
        self.last_hidden_size = dense_dim
        self.num_queries = num_queries
        # self.image_transformer = T.ToPILImage()

        self.save_hyperparameters()
    
    def forward(self, batch):
        """ Shut up """
        text = batch['text']
        image = batch['image']

        ############################################
        # Obj Detection Start
        ############################################
        # images_list = [self.transform(batch_img) for batch_img in image]

        # od_inputs = self.od_feature_extractor(images=images_list, return_tensors="pt")
        # od_inputs = od_inputs.to(self.device)
        # od_outputs = self.od_model(**od_inputs)
        # # ic(od_outputs.keys())
        # # for key in od_outputs.keys():
        # #     ic(key, od_outputs[key].shape)

        # # ic(od_outputs.last_hidden_state.shape)
        # image_x = od_outputs.last_hidden_state

        # image_x = image_x.view(image_x.shape[0], 1, -1)
        # image_x = self.od_fc(image_x)
        # image_x = torch.tanh(image_x)
        # image_x = F.dropout(image_x, p=self.dropout_rate)
        # # ic(image_x.shape)


        image = torch.transpose(image, 1, 2)
        image = torch.transpose(image, 2, 3)
        images_list = [cv2.cv2.cvtColor(batch_img.numpy(), cv2.COLOR_RGB2BGR) for batch_img in image]
        # ic()
        image_x = self.detr2.get_visual_embeds(images_list)
        # ic()
        # images_list = [self.image_transformer(x_) for x_ in image]

        # od_inputs = self.od_feature_extractor(images=images_list, return_tensors="pt")
        # od_inputs = od_inputs.to(self.device)
        # od_outputs = self.od_model(**od_inputs)
        # # ic(od_outputs.keys())
        # # for key in od_outputs.keys():
        # #     ic(key, od_outputs[key].shape)

        # image_x = od_outputs.last_hidden_state
        # # ic(image_x.shape)

        # # image_x = image_x.view(image_x.shape[0], 1, -1)
        # image_x = self.od_fc(image_x)
        # image_x = image_x.permute(0, 2, 1)
        # image_x = F.adaptive_avg_pool1d(image_x, 1)
        # image_x = image_x.permute(0, 2, 1)
        # # image_x = torch.squeeze(image_x, dim=-1)
        # image_x = torch.tanh(image_x)
        ############################################
        # Obj Detection End
        ############################################

        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_length)
        inputs = inputs.to(self.device)

        inputs.update(
            {
                "visual_embeds": image_x,
                "visual_token_type_ids": torch.ones(image_x.shape[:-1], dtype=torch.long).to(self.device),
                "visual_attention_mask": torch.ones(image_x.shape[:-1], dtype=torch.float).to(self.device),
            }
        )

        if self.to_freeze:
            with torch.no_grad():
                x = self.visual_bert(**inputs)
        else:
            x = self.visual_bert(**inputs)

        x = x.pooler_output
        x = x.view(x.shape[0], -1)

        x = torch.squeeze(x)

        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate)

        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate)

        if self.include_top:
            x = self.fc3(x)

        x = torch.squeeze(x, dim=1)
        return x


@click.command()
@click.option('--freeze', default=True, help='Freeze models')
@click.option('--lr', default=1e-4, help='Learning rate')
@click.option('--max_length', default=128, help='Max length')
@click.option('--dense_dim', default=256, help='Dense dim')
@click.option('--dropout_rate', default=0.1, help='Dropout rate')
@click.option('--num_queries', default=50, help='Number of queries')
@click.option('--batch_size', default=0, help='Batch size')
# Train kwargs
@click.option('--epochs', default=10, help='Epochs')
@click.option('--model_dir', default='/tmp', help='Save dir')
@click.option('--grad_clip', default=1.0, help='Gradient clip')
@click.option('--fast_dev_run', default=False, help='Fast dev run')
@click.option('--project', default="visual-bert-with-od", help='Project')
def main(freeze, lr, max_length, dense_dim, dropout_rate, num_queries, batch_size,
         **train_kwargs):
    """ train model """

    model = VisualBertWithODModule(
        freeze=freeze,
        lr=lr, 
        max_length=max_length, 
        dense_dim=dense_dim, 
        dropout_rate=dropout_rate,
        num_queries=num_queries,
        batch_size=batch_size,)
    base_train(model=model, batch_size=batch_size, **train_kwargs)


if __name__ == "__main__":
    pl.seed_everything(42)
    main()
