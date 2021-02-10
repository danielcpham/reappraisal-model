import pytorch_lightning as lit
from torch import nn, optim
from torch.nn import functional as F
from transformers import AutoModel


class LightningReapp(lit.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()

        self.lr = lr
        self.save_hyperparameters()

        self.model = AutoModel.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        for p in self.model.parameters():
            p.requires_grad = False

        self.avg = nn.AvgPool1d(150)

        self.classifier = nn.Sequential(
            nn.Linear(768, 50), nn.ReLU(), nn.Linear(50, 10), nn.ReLU()
        )

        # define metrics
        self.loss = lit.metrics.MeanSquaredError()

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids, attention_mask)
        last_hidden_state = output.last_hidden_state
        avg = self.avg(last_hidden_state.transpose(2, 1))
        out = self.classifier(avg.transpose(2, 1)).squeeze()
        return out

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        # destructure batch
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        score = batch["score"]

        # Compute the loss
        output = self(input_ids, attention_mask)
        loss = self.loss(output.sum(dim=1), score)
        self.log('train/loss', self.loss)
        return loss

    # def validation_step(self, batch, batch_idx):
    #     input_ids = batch["input_ids"]
    #     attention_mask = batch["attention_mask"]
    #     score = batch["score"]
    #     output = self(input_ids, attention_mask)
    #     loss = self.loss(output.sum(dim=1), score)
    #     self.log('val/loss', loss)
    #     return loss
