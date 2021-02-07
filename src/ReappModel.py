import torch 
from torch import nn
from torch.nn import functional as F
from transformers import DistilBertForSequenceClassification


class ReappModel(nn.Module):
    def __init__(self) -> None:
        """Generates a sentiment classifier model from a pretrained huggingface model.
        Args:
        """
        D_in, H, D_out = 768, 50, 7
        super(ReappModel, self).__init__()
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english"
        )
        self.bert_encoder = model


        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(H, D_out),
            nn.ReLU(),
            nn.Linear(D_out, 8)
        )
        # Freeze BERT.
        # for param in self.bert_encoder.parameters():
        #     param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        ) # output is of type SequenceClassifierOutput
        # Note: Loss here is the regression for sentiment classifier labels (neg, pos)
        print(bert_output)
        last_hidden_state = bert_output.hidden_states[-1]
        
        # Softmax the logits to get the probability of that label.
        logits = self.classifier(last_hidden_state)
        return logits
