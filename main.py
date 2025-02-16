import pandas as pd
import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from datasets import Dataset
from peft import LoraConfig, get_peft_model

from transformers import TrainingArguments
from trl import SFTTrainer
from peft import PeftModel

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type= 'nf4',
    bnb_4bit_use_double_quant = True,
    bnb_4bit_compute_dtype=torch.bfloat16
)
model_id = 'beomi/gemma-ko-7b'
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config = bnb_config, device_map={"":0})
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')

dataset = Dataset.from_pandas(train)

train_test_split = dataset.train_test_split(test_size=0.1, seed=42)

train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

def create_prompt(input, output):
    prompt = (
        "<start_of_turn> Your task is to transform the given obfuscated Korean review into a clear, correct, and natural-sounding Korean review that reflects its original meaning.\n"
        f"Input: {input}\n"
        "<end_of_turn>\n"
        "<start_of_turn>Assistant:\n"
        f"Output: {output}"
    )
    return prompt


def format_chat_template(row):
    prompt = create_prompt(row["input"], row["output"])
    tokens = tokenizer.encode(prompt, truncation=True, max_length=512)
    row["input_ids"] = tokens
    return row

# 데이터셋에 적용
train_dataset = train_dataset.map(format_chat_template, batched=False)
test_dataset = test_dataset.map(format_chat_template, batched=False)

lora_config = LoraConfig(
    r = 8,
    lora_alpha = 32,
    target_modules = [
    "q_proj", "v_proj", "k_proj", "o_proj", 
    "gate_proj", "down_proj", "up_proj"
],
    lora_dropout = 0.1,
    bias ='none',
    task_type ='CAUSAL_LM'
)

print('데이터 전처리 끝')

model = get_peft_model(model, lora_config)

# 모델을 훈련 모드로 설정
model.train()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 훈련 인자 설정
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    optim="paged_adamw_32bit",
    eval_strategy="steps",
    eval_steps=100, # 모델의 평가 주기
    logging_dir="./logs",
    logging_steps=100,
    warmup_steps=10, # 학습률 스케줄링
    logging_strategy="steps",
    learning_rate=2e-4,
    group_by_length=True,
    fp16=True
)
 
# Trainer 초기화 및 훈련
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    args=training_args,
    peft_config=lora_config,
    formatting_func=lambda x: tokenizer.decode(x['input_ids'])

)

#  파인튜닝 시작
trainer.train()

ADAPTER_MODEL = "lora_adapter_7b"

trainer.model.save_pretrained(ADAPTER_MODEL)

BASE_MODEL = "beomi/gemma-ko-7b"

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map='auto', torch_dtype=torch.float16)

model = PeftModel.from_pretrained(model, ADAPTER_MODEL, device_map='auto', torch_dtype=torch.float16)

model.save_pretrained('gemma_7b_finetuning')

FINETUNE_MODEL = "./gemma_7b_finetuning"

print('학습 끝')

finetune_model = AutoModelForCausalLM.from_pretrained(FINETUNE_MODEL, device_map={"":0})
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

from transformers import pipeline

# 텍스트 생성 파이프라인 초기화
text_gen_pipeline = pipeline(
    "text-generation",
    model=finetune_model,
    tokenizer=tokenizer,
)

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


batch_size = 16  # Adjust based on GPU memory
prompts = [
    f"<start_of_turn> Your task is to transform the given obfuscated Korean review into a clear, correct, and natural-sounding Korean review that reflects its original meaning.\n"
    f"Input: {query}\n"
    "<end_of_turn>\n"
    "<start_of_turn>Assistant:\n"
    "Output:"
    for query in test['input']
]

restored_reviews = []
for i in range(0, len(prompts), batch_size):
    batch_prompts = prompts[i:i + batch_size]
    
    generated_texts = text_gen_pipeline(
        batch_prompts,
        num_return_sequences=1,
        temperature=0.2,
        top_p=0.9,
        max_new_tokens=100,  # Adjust based on expected output length
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id
    )
    
    for generated in generated_texts:
        generated_text = generated[0]['generated_text']
        output_start = generated_text.find("Output:")
        restored_reviews.append(generated_text[output_start + len("Output:"):].strip() if output_start != -1 else generated_text.strip())

# Save results
submission['output'] = restored_reviews
submission['output'] = submission['output'].apply(lambda x: x.split("<end_of_turn>")[0])
submission.to_csv('./submission.csv', index=False, encoding='utf-8-sig')
