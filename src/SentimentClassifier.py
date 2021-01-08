from torch import nn
from transformers import PreTrainedModel, DistilBertModel


# Generic test sentiment classifier
# TODO: Extend to an objective trainer somehow

class SentimentClassifier(nn.Module):
    def __init__(self, n_classes: int, model_class: DistilBertModel, model_name: str):
        """Generates a sentiment classifier model from a pretrained huggingface model.

        Args:
            n_classes (int): [description]
            model_class (Model): [description]
            model_name (str): [description]
        """
        super(SentimentClassifier, self).__init__()
        self.bert = model_class.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.3) # Dropout: randomly zero some of the elements with probability p
        # Simple feedforward network from the last hidden state to the output (the classification labels)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
        # Multiclass logistic regression
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
