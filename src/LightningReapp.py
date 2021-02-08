import pytorch_lightning as lit
from torch import nn, optim
from torch.nn import functional as F
from transformers import AutoModel


class LightningReapp(lit.LightningModule):
  def __init__(self):
    super().__init__()
    self.model = AutoModel.from_pretrained('distilbert-base-cased')
    for p in self.model.parameters():
        p.requires_grad = False
    self.classifier = nn.Sequential(
      nn.Linear(768, 50),
      nn.ReLU(),
      #nn.Dropout(0.5),
      nn.Linear(50, 10),
      nn.ReLU()
    )

  def forward(self, input_ids, attention_mask):
    output = self.model(input_ids, attention_mask)
    last_hidden_state = output.last_hidden_state
    return self.classifier(last_hidden_state)

  def configure_optimizers(self):
    optimizer = optim.Adam(self.parameters(), lr=1e-3)
    return optimizer

  def training_step(self, batch, batch_idx):
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    score = batch['score']
    output = self(input_ids, attention_mask)
    loss = F.mse_loss(output.sum(), score)
    return loss


