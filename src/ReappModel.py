from torch import nn
from transformers import PreTrainedModel


class ReappModel(nn.Module):
    def __init__(self, pretrained_model : PreTrainedModel):
        """Generates a sentiment classifier model from a pretrained huggingface model.
        Args:
        """
        super(ReappModel, self).__init__()
        self.bert = pretrained_model

        config = self.bert.config
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, config.num_labels)
        self.dropout = nn.Dropout(config.seq_classif_dropout)
        # Freeze the classifier
        # self.classifier.requires_grad(False)
        self.num_labels = config.num_labels
        #Use linear layer to the number of "labels" or "features" to fit to score, and then logsoftmax 
        self.reapp_scorer = nn.Linear(config.num_labels, 1)
        self.loss = nn.MSELoss()


    def classify(self, inputs):
        # batchsize, sequence length, dimension of vectors
        hidden_state = inputs[0] # (bs, seq_len, dim)
        pooled_output = hidden_state[:,0] # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output) # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output) # (bs, dim)
        pooled_output = self.dropout(pooled_output) # (bs, dim)
        logits = self.classifier(pooled_output)
        predicted = self.reapp_scorer(logits)
        return predicted

    def forward(self, input_ids, attention_mask, score, **bert_kwargs):
        # pooled output applies the activation function on the first token's hidden state
        # Returns last_hidden_state, hidden_states (opt), attentions (opt)
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **bert_kwargs)
        # Manually attach the classifier head
         # (bs, num_labels)
        predicted = self.classify(bert_output)
        predicted = self.reapp_scorer(predicted)
        loss = self.loss(predicted.float(), score.float())
        print(predicted)
        print(loss)
        return loss, bert_output