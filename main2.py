import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 로드
train = pd.read_csv('./train.csv', encoding='utf-8-sig')
test = pd.read_csv('./test.csv', encoding='utf-8-sig')

# 커스텀 데이터셋
class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt = self.data.iloc[idx]["input"]
        response = self.data.iloc[idx]["output"]

        additional_prompt = (
            "Restore the following obfuscated Korean text to its original, clear, and natural form.\n"
        )

        combined = f"{additional_prompt}\nObfuscated Text: {prompt}\nRestored Text: {response}"

        tokens = self.tokenizer(
            combined,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = self.tokenizer(
            response,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )["input_ids"].squeeze()

        return {
            "input_ids": tokens["input_ids"].squeeze(),
            "attention_mask": tokens["attention_mask"].squeeze(),
            "labels": labels
        }

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

dataset = CustomDataset(train, tokenizer)

# LoRA 설정 수정
lora_config = LoraConfig(
    r=8,  # LoRA rank 증가
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj"],
    task_type="CAUSAL_LM",
)

# 모델 로드
BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16)
model = model.to(device)

# LoRA 적용
model = get_peft_model(model, lora_config)

# 학습 설정
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    num_train_epochs=15,
    logging_dir="./logs",
    save_steps=500,
    save_total_limit=5,
    evaluation_strategy="no",
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

# 학습 실행
trainer.train()

# 모델 저장
model.save_pretrained("./trained_model")
tokenizer.save_pretrained("./trained_model")

# 모델 평가 모드
model.eval()

# 테스트 데이터셋
test_dataset = CustomDataset(test, tokenizer)
predictions = []

# 추론
for i in range(len(test_dataset)):
    input_ids = test_dataset[i]["input_ids"].unsqueeze(0).to(device)
    attention_mask = test_dataset[i]["attention_mask"].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=256,  
            num_beams=1, 
            early_stopping=True
        )

    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    predictions.append(decoded_output)

# 제출 파일 저장
submission = pd.read_csv('./sample_submission.csv', encoding='utf-8-sig')
submission["output"] = predictions
submission.to_csv('./result.csv', index=False, encoding='utf-8-sig')
