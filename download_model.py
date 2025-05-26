from transformers import AutoModel, AutoTokenizer

model_name = "microsoft/deberta-v3-large"

# Download and cache model + tokenizer
AutoTokenizer.from_pretrained(model_name)
AutoModel.from_pretrained(model_name)
