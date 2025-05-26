import pandas as pd
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, DebertaV2Tokenizer
import click
import torch
from torch.nn.functional import softmax

class CFG:
    model_name_or_path = "Shushant/panclef_data_deberta_finetuned"
    max_length = 128
    batch_size = 32
    output_file = "outputs.jsonl"

class Predictor:
    def __init__(self):
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(CFG.model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(CFG.model_name_or_path)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def load_data(self, file_path: str) -> pd.DataFrame:
        df = pd.read_json(file_path, lines=True)
        if 'id' not in df.columns or 'text' not in df.columns:
            raise ValueError("Expected columns: 'id' and 'text'")
        return df

    def tokenize(self, texts):
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=CFG.max_length,
            return_tensors="pt"
        )

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        all_probs = []

        for i in range(0, len(df), CFG.batch_size):
            batch = df.iloc[i:i+CFG.batch_size]
            encodings = self.tokenize(batch["text"].tolist())
            encodings = {k: v.to(self.device) for k, v in encodings.items()}

            with torch.no_grad():
                outputs = self.model(**encodings)
                probs = softmax(outputs.logits, dim=-1)[:, 1]  # probability of class 1
                all_probs.extend(probs.cpu().numpy())

        predictions_df = pd.DataFrame({
            "id": df["id"],
            "label": all_probs
        })

        return predictions_df


@click.command()
@click.option('--input_dir', required=True)
@click.option('--output', required=True)
def main(input_dir, output):
    predictor = Predictor()
    df = predictor.load_data(input_dir)
    preds = predictor.predict(df)
    preds.to_json(output, orient="records", lines=True)

if __name__ == "__main__":
    main()

