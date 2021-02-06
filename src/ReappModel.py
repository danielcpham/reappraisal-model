from torch import nn
from transformers import DistilBertForSequenceClassification



class ReappModel(nn.Module):
    def __init__(self):
        """Generates a sentiment classifier model from a pretrained huggingface model.
        Args:
        """
        super(ReappModel, self).__init__()
        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

        self.num_labels = model.config.num_labels

        self.transformer = nn.Sequential(*list(model.children())[:-3])
        # Freeze sentiment classification since it's already been finetuned
        self.classifier_head = nn.Sequential(*list(model.children())[-3:])
        self.classifier_head.requires_grad_(False)

        #Use linear layer to the number of "labels" or "features" to fit to score, and then logsoftmax 
        self.reapp_scorer = nn.Linear(model.config.num_labels, 1)
        self.loss = nn.MSELoss()


    def forward(self, input_ids, attention_mask, score, **bert_kwargs):
        # pooled output applies the activation function on the first token's hidden state
        # Returns last_hidden_state, hidden_states (opt), attentions (opt)
        bert_output = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **bert_kwargs)
        predicted
        loss = self.loss(predicted.float(), score.float().unsqueeze(-1))
        return loss, bert_output