
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping

import transformers

from models.baseline import BaseMaeMaeModel
from hateful_memes.data.hateful_memes import MaeMaeDataset
from hateful_memes.data.hateful_memes import MaeMaeDataModule
from utils import get_project_logger

class BaseTextMaeMaeModel(BaseMaeMaeModel):
    def __init__(
        self, 
        lr=0.003, 
        vocab_size=256, 
        embed_dim=512, 
        dense_dim=128, 
        max_length=128,
        batch_size=32):
        super().__init__()

        
        # self.pipeline = transformers.pipeline("feature-extraction", framework='pt')
        # transformers.Pretrainged
        # self.pipeline = transformers.pipeline('feature-extraction', model='bert-base-cased', tokenizer='bert-base-cased')
        # self.pipeline = transformers.pipeline('text-classification', 'Hate-speech-CNERG/bert-base-uncased-hatexplain')
        # self.pipeline = transformers.pipeline('text-classification', 'Hate-speech-CNERG/bert-base-uncased-hatexplain')
        # self.tokenizer = transformers.AutoTokenizer.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain")
        # self.modelm = transformers.AutoModel.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain")
        # TODO could fine tune
        # self.tokenizer

        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.lr = lr

        self.l1 = nn.Linear(embed_dim, dense_dim)
        self.l2 = nn.Linear(dense_dim, 1)
        # TODO consider 3 classes for offensive detection

        # self.log("batch_size", batch_size)
        self.batch_size = batch_size
        self.max_length = max_length

        self.save_hyperparameters()
    
    # def preprocess(self, x_img, x_txt):
    #     x_txt = self.tokenizer(x_txt, return_tensors='pt', padding='max_length', truncation=True)
    #     return x_img, x_txt
    #     # return super().preprocess(x_img, x_txt)
    

    def forward(self, batch):
        text_features = batch['text_features']
        text_offset = batch['text_offset']
        # x = x_txt
        # ic(text_features.shape)
        # ic(text_offset.shape)
        x = self.embedding(text_features, text_offset)
        # ic(x.shape)
        # ic(len(x))
        # ic(x)
        # x = self.pipeline(x, max_length=512, padding='max_length', truncation=True, return_tensors='pt')

        # for key, value in x.items():
        #     x[key] = value.to(self.device)
        # x = self.modelm(**x)
        # ic(x['pooler_output'].shape)
        # ic(x['last_hidden_state'].shape)
        # x = x['pooler_output']
        # ic(x.shape)
        x = self.l1(x)
        # ic(x.shape)
        x = F.relu(x)
        x = self.l2(x)
        # ic(x.shape)
        x = torch.sigmoid(x)
        # x = self.pipeline(x)
        # ic(len(x))
        # ic(len(x[0]))
        # ic(len(x[0]))
        # ic(len(x[3]))
        # ic(x)
        # x = torch.Tensor(x)
        # ic(x.shape)
        # ic(x.shape)
        # x = self.l1(x)
        # ic(x.shape)
        # return [i['score'] for i in x]
        x = torch.squeeze(x)
        return x

def create_tokinizer(data: torch.utils.data.Dataset):
    # # Construction 1
    # from spacy.tokenizer import Tokenizer
    # from spacy.lang.en import English
    # nlp = English()
    # # Create a blank Tokenizer with just the English vocab
    # tokenizer = Tokenizer(nlp.vocab)

    # Construction 2
    from spacy.lang.en import English
    nlp = English()
    # Create a Tokenizer with the default settings for English
    # including punctuation rules and exceptions
    tokenizer = nlp.tokenizer


# Model to process text
@click.command()
@click.option('--batch_size', default=32, help='Batch size')
@click.option('--lr', default=1e-4, help='Learning rate')
@click.option('--dense_dim', default=256, help='Dense dim')
@click.option('--grad_clip', default=1.0, help='Gradient clipping')
@click.option('--dropout_rate', default=0.1, help='Dropout rate')
@click.option('--epochs', default=100, help='Epochs')
@click.option('--model_dir', default='/tmp', help='Model path')
@click.option('--fast_dev_run', type=bool, default=False, help='Fast dev run')
@click.option('--log_dir', default="data/08_reporting/simple_mlp_image", help='Fast dev run')
@click.option('--project', default="simple-mlp-image", help='Fast dev run')
@click.option('--monitor_metric', default="val/loss", help='Metric to monitor')
@click.option('--monitor_metric_mode', default="min", help='Min or max')
def main(batch_size, lr, dense_dim, grad_clip,
         dropout_rate, epochs, model_dir, fast_dev_run,
         log_dir, project, monitor_metric, monitor_metric_mode):

    """ Train Text model """
    logger = get_project_logger(project=project, save_dir=log_dir, offline=fast_dev_run)

    checkpoint_callback = ModelCheckpoint(monitor="val/acc", mode="max", dirpath="data/06_models/hateful_memes", save_top_k=1)
    early_stopping = EarlyStopping(
            monitor=monitor_metric,
            patience=10, 
            mode=monitor_metric_mode,
            verbose=True)

    trainer = Trainer(
        devices=1, 
        accelerator='auto',
        max_epochs=100, 
        logger=logger, 
        gradient_clip_val=1.0,
        callbacks=[checkpoint_callback, early_stopping])
    
    dataset = MaeMaeDataset(
        "data/01_raw/hateful_memes",
        train=True,
    )
    train_dataset = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        collate_fn=MaeMaeDataset.collate_batch,

    )
    model = BaseTextMaeMaeModel(
         vocab_size=len(dataset.vocab),
         batch_size=32
    )
    trainer.fit(model, datamodule=MaeMaeDataModule(batch_size=32))


if __name__ == "__main__":
    main()