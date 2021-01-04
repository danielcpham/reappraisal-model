from torch import nn


# Generic tester sentiment classifier
class SentimentClassifier(nn.Module):
    def __init__(self, n_classes: int):
        super(SentimentClassifier, self).__init__()
        self.bert = DistilBertModel.from_pretrained(PRETRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3) 
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes) # Linear feedforward networks
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