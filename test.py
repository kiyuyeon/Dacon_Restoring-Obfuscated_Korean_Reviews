import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset

# 테스트 데이터 로드
test = pd.read_csv('./test.csv', encoding='utf-8-sig')

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 및 토크나이저 로드
model = AutoModelForCausalLM.from_pretrained("./trained_model", torch_dtype=torch.float16).to(device)
tokenizer = AutoTokenizer.from_pretrained("./trained_model")

# 모델 평가 모드
model.eval()

# 커스텀 데이터셋 클래스 정의
class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt = self.data.iloc[idx]["input"]

        additional_prompt = "Restore the following obfuscated Korean text to its original, clear, and natural form.\n"
        combined = f"{additional_prompt}\nObfuscated Text: {prompt}\nRestored Text:"

        tokens = self.tokenizer(
            combined,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": tokens["input_ids"].squeeze(),
            "attention_mask": tokens["attention_mask"].squeeze(),
        }

# 테스트 데이터셋 생성
test_dataset = CustomDataset(test, tokenizer)
predictions = []

# 추론 실행
for i in range(len(test_dataset)):
    input_ids = test_dataset[i]["input_ids"].unsqueeze(0).to(device)
    attention_mask = test_dataset[i]["attention_mask"].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=256,
            num_beams=1,
            early_stopping=True
        )

    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    predictions.append(decoded_output)

# 제출 파일 저장
submission = pd.read_csv('./sample_submission.csv', encoding='utf-8-sig')
submission["output"] = predictions
submission.to_csv('./result.csv', index=False, encoding='utf-8-sig')
