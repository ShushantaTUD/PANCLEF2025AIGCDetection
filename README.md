# Submission for PAN CLEF 2025 generative ai text detection

The trained models is in the model directory. 
The predictions are in the prediction.jsonl file.


## Submission to TIRA
1. Check if the code works:
  ```bash
  tira-cli code-submission --dry-run --path ./ --task generative-ai-authorship-verification-panclef-2025 --dataset pan25-generative-ai-detection-smoke-test-20250428-training --mount-hf-model microsoft/deberta-v3-large
  ```

2. If that ran successfully, you can omit the `-dry-run` argument to submit:
  ```bash
  tira-cli code-submission --path ./ --task generative-ai-authorship-verification-panclef-2025 --dataset pan25-generative-ai-detection-smoke-test-20250428-training --mount-hf-model microsoft/deberta-v3-large
  ```

## Huggingface 
The model is available on Huggingface platform through 
https://huggingface.co/Shushant/panclef_data_deberta_finetuned
