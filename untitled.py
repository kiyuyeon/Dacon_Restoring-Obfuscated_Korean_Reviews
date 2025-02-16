import pandas as pd
import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from datasets import Dataset
from peft import LoraConfig, get_peft_model

from transformers import TrainingArguments
from trl import SFTTrainer
from peft import PeftModel

# model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type= 'nf4',
    bnb_4bit_use_double_quant = True,
    bnb_4bit_compute_dtype=torch.bfloat16
)
model_id = 'beomi/gemma-ko-7b'
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="cuda:0"  # 단일 GPU 사용
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

# data
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')

text_gen_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)


# 프롬프트 생성 및 텍스트 생성
restored_reviews = []

for index, row in test.iloc[:10].iterrows():
    query = row['input']
    prompt = (
        "<start_of_turn> Your task is to transform the given obfuscated Korean review into a clear, correct, and natural-sounding Korean review that reflects its original meaning.\n"
        f"Input: {query}\n"
        "<end_of_turn>\n"
        "<start_of_turn>Assistant:\n"
        "Output:"
    )

    # 텍스트 생성
    generated = text_gen_pipeline(
        prompt,
        num_return_sequences=1,
        temperature=0.2,
        top_p=0.9,
        max_new_tokens=len(query),
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id
    )

    # 생성된 텍스트에서 출력 부분 추출
    generated_text = generated[0]['generated_text']
    output_start = generated_text.find("Output:")
    if output_start != -1:
        restored_reviews.append(generated_text[output_start + len("Output:"):].strip())
    else:
        restored_reviews.append(generated_text.strip())


# 데이터셋 준비
dataset = Dataset.from_pandas(train)
train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

# 프롬프트 생성 함수
def create_prompt(input_text, output_text):
    return (
        "<start_of_turn> Your task is to transform the given obfuscated Korean review into a clear, correct, and natural-sounding Korean review that reflects its original meaning.\n"
        f"Input: {input_text}\n"
        "<end_of_turn>\n"
        "<start_of_turn>Assistant:\n"
        f"Output: {output_text}"
    )

def format_data(examples):
    inputs = examples['input']
    outputs = examples['output']
    prompts = [create_prompt(input_text, output_text) for input_text, output_text in zip(inputs, outputs)]
    return {'prompt': prompts}

train_dataset = train_dataset.map(format_data, batched=True, num_proc=4)
test_dataset = test_dataset.map(format_data, batched=True, num_proc=4)

# # 데이터셋을 GPU로 이동
# train_dataset = train_dataset.map(lambda x: {'input_ids': x['input'].to('cuda:0')})
# test_dataset = test_dataset.map(lambda x: {'input_ids': x['input'].to('cuda:0')})

# LoRA 설정
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
    lora_dropout=0.1,
    bias='none',
    task_type='CAUSAL_LM'
)
model = get_peft_model(model, lora_config)


# 모델을 훈련 모드로 설정
model.train()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# 훈련 인자 설정
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    optim="paged_adamw_32bit",
    eval_strategy="steps",
    eval_steps=100,
    logging_dir="./logs",
    logging_steps=100,
    warmup_steps=10,
    learning_rate=2e-4,
    group_by_length=True,
    fp16=True
)
 
# 훈련 실행
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    args=training_args,
    peft_config=lora_config,
    formatting_func=lambda x: x['input']
)
trainer.train()


# 모델 저장 및 로드
ADAPTER_MODEL = "lora_adapter_7b"
trainer.model.save_pretrained(ADAPTER_MODEL)

BASE_MODEL = "beomi/gemma-ko-7b"
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map='cuda:0', torch_dtype=torch.float16)
model = PeftModel.from_pretrained(model, ADAPTER_MODEL, device_map='cuda:0', torch_dtype=torch.float16)
model.save_pretrained('gemma_7b_finetuning')

# 테스트 데이터 예측
FINETUNE_MODEL = "./gemma_7b_finetuning"
finetune_model = AutoModelForCausalLM.from_pretrained(FINETUNE_MODEL, device_map={"":0})
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

text_gen_pipeline = pipeline("text-generation", model=finetune_model, tokenizer=tokenizer)

restored_reviews = []
for index, row in test.iterrows():
    query = row['input']
    prompt = (
        "<start_of_turn> Your task is to transform the given obfuscated Korean review into a clear, correct, and natural-sounding Korean review that reflects its original meaning.\n"
        f"Input: {query}\n"
        "<end_of_turn>\n"
        "<start_of_turn>Assistant:\n"
        "Output:"
    )

    generated = text_gen_pipeline(
        prompt,
        num_return_sequences=1,
        temperature=0.2,
        top_p=0.9,
        max_new_tokens=len(query),
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id
    )

    generated_text = generated[0]['generated_text']
    output_start = generated_text.find("Output:")
    if output_start != -1:
        restored_reviews.append(generated_text[output_start + len("Output:"):].strip())
    else:
        restored_reviews.append(generated_text.strip())

# 결과 저장
submission = pd.read_csv('./sample_submission.csv', encoding='utf-8-sig')
submission['output'] = restored_reviews
submission['output'] = submission['output'].apply(lambda x: x.split("<end_of_turn>")[0])
submission.to_csv('./submission.csv', index=False, encoding='utf-8-sig')