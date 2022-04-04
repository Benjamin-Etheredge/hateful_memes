from matplotlib.pyplot import autoscale
import pytorch_lightning as pl
import torch
import click
from transformers import BertTokenizer, VisualBertModel
import torchvision.models as models
from hateful_memes.data.hateful_memes import MaeMaeDataModule
from torch.nn import functional as F
from torch import nn
from dvclive.lightning import DvcLiveLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from icecream import ic
# ic.disable()

# model = VisualBertModel.from_pretrained("uclanlp/visualbert-vqa-coco-pre")
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# inputs = tokenizer("What is the man eating?", return_tensors="pt")
# # this is a custom function that returns the visual embeddings given the image path
# # visual_embeds = get_visual_embeddings(image_path)
# visual_embeds = torch.rand((1, 2048)).unsqueeze(0)

# visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
# visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)
# inputs.update(
#     {
#         "visual_embeds": visual_embeds,
#         "visual_token_type_ids": visual_token_type_ids,
#         "visual_attention_mask": visual_attention_mask,
#     }
# )
# ic(inputs)
# outputs = model(**inputs)
# last_hidden_state = outputs.last_hidden_state
# model = models.vit_b_16(pretrained=True)
# ic(model)
# model = models.vit_l_16(pretrained=True)
# ic(model)
# ic(models.alexnet(pretrained=True))
# ic(models.alexnet(pretrained=True))



class VisualBertModule(pl.LightningModule):

    def __init__(
        self, 
        lr=0.003, 
        max_length=512, 
        include_top=True,
        dropout_rate=0.0,
        dense_dim=256,
    ):
        super().__init__()
        # self.hparams = hparams
        self.model = VisualBertModel.from_pretrained("uclanlp/visualbert-vqa-coco-pre")
        # for param in self.model.parameters():
        #     param.requires_grad = False
        ic(self.model)
        # model.
        # self.model.freeze()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        # self.resnet18 = models.resnet18(pretrained=True)
        # resnet = models.resnet152(pretrained=True)
        resnet = models.resnet50(pretrained=True)
        # ic(resnet)
        self.num_ftrs_resnet = resnet.fc.in_features
        # for param in resnet.parameters():
        #     param.requires_grad = False
        resnet.fc = nn.Flatten()
        ic(resnet)
        self.resnet = resnet

        # self.alexnet = models.alexnet(pretrained=True)
        # ic(self.alexnet)
        # self.num_ftrs_alexnet = self.alexnet.classifier[6].in_features
        # for param in resnet.parameters():
        #     param.requires_grad = False
        
        # inception = models.inception_v3(pretrained=True)

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
        # self.resnet18.freeze()

        self.save_hyperparameters()
    
    def _shared_step(self, batch):
        y_hat = self.forward(batch)
        y = batch['label']
        # loss = F.binary_cross_entropy(y_hat, y.to(y_hat.dtype))
        # acc = torch.sum(torch.round(y_hat) == y.data) / (y.shape[0] * 1.0)
        loss = F.binary_cross_entropy_with_logits(y_hat, y.to(y_hat.dtype))
        acc = torch.sum(torch.round(torch.sigmoid(y_hat)) == y.data) / (y.shape[0] * 1.0)
        return loss, acc

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch['image'].size(0))
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch['image'].size(0))
        return loss
    
    def training_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch['image'].size(0))
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch['image'].size(0))
        return loss


    
    def forward(self, batch):
        text = batch['text']
        image = batch['image']
        image_x = self.resnet(image)
        # with torch.no_grad():
            # image_x = self.resnet(image)
        # ic(image_x.shape)
        image_x = image_x.view(image_x.shape[0], -1)

        # ic(image_x.shape)
        # image_x = self.fc(image_x)
        # image_x = F.relu(image_x)
        # ic(image_x.shape)
        image_x = image_x.unsqueeze(1)

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
        # for key,value in inputs.items():
        #     ic(key, value.shape)


        # ic(inputs)
        # with torch.no_grad():
        x = self.model(**inputs)
        x = x.pooler_output
        x = x.view(x.shape[0], -1)
        # ic(x.shape)

        # ic(x.last_hidden_state.shape)
        # ic(x.pooler_output.shape)


        # x = x.mean(dim=1)
        # x = x.unsqueeze(1)
        # ic(x.shape)
        # ic(x.shape)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(input=x, p=self.dropout_rate)

        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(input=x, p=self.dropout_rate)
        # ic(x.shape)
        x = self.fc3(x)
        x.squeeze_()
        # ic(x.shape)
        # x = torch.sigmoid(x)
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)



@click.command()
@click.option('--batch_size', default=32, help='Batch size')
@click.option('--lr', default=1e-4, help='Learning rate')
@click.option('--max_length', default=128, help='Max length')
@click.option('--dense_dim', default=256, help='Dense dim')
@click.option('--dropout_rate', default=0.1, help='Dropout rate')
@click.option('--epochs', default=10, help='Epochs')
@click.option('--model_dir', default='/tmp', help='Save dir')
@click.option('--gradient_clip_value', default=1.0, help='Gradient clip')
@click.option('--fast_dev_run', default=False, help='Fast dev run')
def main(batch_size, lr, max_length, dense_dim, dropout_rate, 
         epochs, model_dir, gradient_clip_value, fast_dev_run):

    logger = DvcLiveLogger() if not fast_dev_run else DvcLogger()
    checkpoint_callback = ModelCheckpoint(
        monitor="val/acc", 
        mode="max", 
        dirpath=model_dir, 
        filename="{epoch}-{step}-{val_acc:.4f}",
        verbose=True,
        save_top_k=1)
    early_stopping = EarlyStopping(
            monitor='val/acc', 
            patience=10, 
            mode='max', 
            verbose=True)

    trainer = pl.Trainer(
        gpus=1, 
        max_epochs=epochs, 
        logger=logger,
        # logger=wandb_logger, 
        gradient_clip_val=gradient_clip_value,
        callbacks=[checkpoint_callback, early_stopping],
        fast_dev_run=fast_dev_run,
        # detect_anomaly=True, # TODO explore more
        # callbacks=[checkpoint_callback])
        # precision=16,
        # auto_scale_batch_size=True,
    )
    
    model = VisualBertModule(
        lr=lr, 
        max_length=max_length, 
        dense_dim=dense_dim, 
        dropout_rate=dropout_rate)
    trainer.fit(
        model, 
        datamodule=MaeMaeDataModule(batch_size=batch_size))


if __name__ == "__main__":
    pl.seed_everything(42)
    main()
    # wandb_logger = WandbLogger(project="Hateful_Memes_Base_Image", log_model=True)
    # checkpoint_callback = ModelCheckpoint(monitor="val/acc", mode="max", dirpath="data/06_models/hateful_memes", save_top_k=1)
