import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW, BitsAndBytesConfig
from torch.utils.data import Dataset, DataLoader
import torch

# # 데이터셋 클래스 정의 (토크나이징 처리)
# class TextDataset(Dataset):
#     def __init__(self, dataframe, tokenizer, max_length=512):
#         self.dataframe = dataframe
#         self.tokenizer = tokenizer
#         self.max_length = max_length

#     def __len__(self):
#         return len(self.dataframe)

#     def __getitem__(self, idx):
#         input_text = self.dataframe['input'][idx]
#         output_text = self.dataframe['output'][idx]
        
#         # 입력과 출력에 대해 토크나이징하고, labels를 출력값으로 설정
#         encoding = self.tokenizer(
#             input_text, output_text, 
#             truncation=True, 
#             padding='max_length', 
#             max_length=self.max_length, 
#             return_tensors="pt"
#         )
        
#         # labels 필드 추가
#         encoding['labels'] = encoding['input_ids'].clone()  # 모델에 'input_ids'를 'labels'로 설정

#         return encoding


# train_df = pd.read_csv('./train.csv', encoding = 'utf-8-sig')

# tokenizer 로드
tokenizer = AutoTokenizer.from_pretrained("beomi/gemma-ko-7b")

# # DataLoader 준비
# train_dataset = TextDataset(train_df, tokenizer)
# train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# BitsAndBytes 4bit 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 모델 로드
model_id = "beomi/gemma-ko-7b"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

# 옵티마이저 설정
optimizer = AdamW(model.parameters(), lr=5e-5)

# # 파인튜닝 학습 루프
# model.train()  # 모델을 학습 모드로 설정

# for epoch in range(3):  # 에폭 수 설정
#     for batch in train_dataloader:
#         input_ids = batch['input_ids'].squeeze(1).to(model.device)  # 입력 아이디
#         labels = batch['labels'].squeeze(1).to(model.device)  # 출력 아이디

#         optimizer.zero_grad()  # 이전 기울기 초기화

#         # 모델에 입력 데이터를 넣고 출력 계산
#         outputs = model(input_ids=input_ids, labels=labels)
#         loss = outputs.loss  # 손실 계산

#         # 역전파
#         loss.backward()

#         # 파라미터 업데이트
#         optimizer.step()

#         print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# # 파인튜닝된 모델 저장
# model.save_pretrained("./fine_tuned_model")

# # 토크나이저 저장
# tokenizer.save_pretrained("./fine_tuned_tokenizer")

# 데이터셋 클래스 수정
class TextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512, is_test=False):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test  # 테스트용 데이터셋 여부 추가

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        input_text = self.dataframe['input'][idx]
        
        if self.is_test:
            # 테스트용 데이터셋에서는 output을 사용하지 않음
            encoding = self.tokenizer(
                input_text, 
                truncation=True, 
                padding='max_length', 
                max_length=self.max_length, 
                return_tensors="pt"
            )
            encoding['labels'] = encoding['input_ids'].clone()  # 모델에 'input_ids'를 'labels'로 설정
        else:
            # 훈련용 데이터셋에서 'output' 사용
            output_text = self.dataframe['output'][idx]
            encoding = self.tokenizer(
                input_text, output_text, 
                truncation=True, 
                padding='max_length', 
                max_length=self.max_length, 
                return_tensors="pt"
            )
            encoding['labels'] = encoding['input_ids'].clone()  # 모델에 'input_ids'를 'labels'로 설정

        return encoding

# 테스트 데이터셋 로드 시 수정
test_df = pd.read_csv('./test.csv')  # 테스트 데이터셋 로드
test_dataset = TextDataset(test_df, tokenizer, is_test=True)  # is_test=True로 설정
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)


# 예측 결과 생성
model.eval()  # 평가 모드로 설정
restored_reviews = []

with torch.no_grad():
    for batch in test_dataloader:
        input_ids = batch['input_ids'].squeeze(1).to(model.device)
        outputs = model.generate(input_ids=input_ids, max_length=600)
        predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        restored_reviews.extend(predictions)

# 제출 파일 생성
submission = pd.read_csv('./sample_submission.csv', encoding='utf-8-sig')
submission['output'] = restored_reviews

# 결과를 파일로 저장
submission.to_csv('./baseline_submission.csv', index=False, encoding='utf-8-sig')
