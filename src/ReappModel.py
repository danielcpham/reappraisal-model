from torch import nn
from transformers import DistilBertModel


class ReappModel(nn.Module):
    def __init__(self, model_class=DistilBertModel, model_name='distilbert-base-cased'):
        """Generates a sentiment classifier model from a pretrained huggingface model.
        Args:
        """
        super(ReappModel, self).__init__()
        self.bert = model_class.from_pretrained(model_name)
        self.loss = nn.MSELoss()
        self.out = nn.ReLU()

    def forward(self, input_ids, attention_mask, score, **bert_kwargs):
        # pooled output applies the activation function on the first token's hidden state
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **bert_kwargs)
        out = self.out(outputs.last_hidden_state)
        print(out.sum((1,2)).shape, score[0].shape)
        loss = self.loss(out.sum((1,2)).float(), score[0].float())
        # Add regression classes for 
        #TODO: we simply sum up the elements here, let's do something smarter?
        return loss, out



