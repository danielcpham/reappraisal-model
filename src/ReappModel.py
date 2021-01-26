from torch import nn
from transformers import DistilBertModel


class ReappModel(nn.Module):
    def __init__(self, model_class=DistilBertModel, model_name='distilbert-base-cased'):
        """Generates a sentiment classifier model from a pretrained huggingface model.

        Args:
            n_classes (int): [description]
            model_class (Model): [description]
            model_name (str): [description]
        """
        super(ReappModel, self).__init__()
        self.bert = model_class.from_pretrained(model_name)
        # Dropout: randomly zero some of the elements with probability p
        self.drop = nn.Dropout(p=0.3) 
        # Simple feedforward network from the last hidden state to the output (the classification labels)
        self.out = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, **bert_kwargs):
        # pooled output applies the activation function on the first token's hidden state
        # Bert Pooler uses tanh activation
        last_hidden_state, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **bert_kwargs)
        output = self.drop(pooled_output)
        output = self.out(output)
        return self.softmax(output)

