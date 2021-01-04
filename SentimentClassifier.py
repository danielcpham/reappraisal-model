from torch import nn
from transformers import Model


# Generic test sentiment classifier
class SentimentClassifier(nn.Module):
    def __init__(self, n_classes: int, model_class: Model, model_name: str):
        """Generates a sentiment classifier model from a pretrained huggingface model.

        Args:
            n_classes (int): [description]
            model_class (Model): [description]
            model_name (str): [description]
        """
        super(SentimentClassifier, self).__init__()

        self.bert = model_class.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.3)
        # Linear feedforward networks
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        # last_hidden_state of the last encoder unit in the bert model.
        # pooled output applies the activation function on the first token's hidden state
        # Bert Pooler uses tanh activation
        last_hidden_state, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask)
        output = self.drop(pooled_output)
        output = self.out(output)
        return self.softmax(output)
