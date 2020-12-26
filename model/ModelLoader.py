from transformers import DistilBertModel, DistilBertTokenizerFast

def load_model(model_name:str):
  # TODO: Make the model generic
  return DistilBertModel.from_pretrained(model_name)

def load_tokenizer(model_name: str):
  # TODO: Make the tokenizer generic
  return DistilBertTokenizerFast.from_pretrained(model_name)

def load(model_name:str = 'distil-bert-uncased'):
  return load_model(model_name), load_tokenizer(model_name)