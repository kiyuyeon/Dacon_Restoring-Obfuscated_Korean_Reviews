import pandas as pd
import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from datasets import Dataset
from peft import LoraConfig, get_peft_model

from transformers import TrainingArguments
from trl import SFTTrainer
from peft import PeftModel
from datasets import Dataset


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type= 'nf4',
    bnb_4bit_use_double_quant = True,
    bnb_4bit_compute_dtype=torch.bfloat16
)
model_id = 'meta-llama/Meta-Llama-3-8B-Instruct'
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
        "<start_of_turn> You are a professional professor of the Korean language. "
        "Your task is to restore an obfuscated Korean review into a clear, natural, and grammatically correct Korean review while preserving its original meaning. "
        "You must strictly adhere to the following conditions:\n"
        "1. The number of sentences in the obfuscated text and the restored review must be exactly the same.\n"
        "2. Interpret the original intent accurately and transform it into a natural sentence without distorting the meaning.\n"
        "3. Ensure that the output is grammatically correct and reads smoothly.\n"
        "4. Avoid excessive changes in word choice while improving clarity.\n"
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


def preprocess_dataset(dataset):
    return dataset.map(lambda x: {"text": tokenizer.decode(x["input_ids"])})
    
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

model = get_peft_model(model, lora_config)

# 모델을 훈련 모드로 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_dataset = preprocess_dataset(train_dataset)
test_dataset = preprocess_dataset(test_dataset)

# 훈련 인자 설정
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=10,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    optim="paged_adamw_32bit",
    eval_strategy="steps",
    eval_steps=100,  # 모델의 평가 주기
    logging_dir="./logs",
    logging_steps=100,
    warmup_steps=10,  # 학습률 스케줄링
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

# 파인튜닝 시작
trainer.train()

sample_text = train_dataset[0]["text"]  # Replace "text" with the correct field
tokenized_output = tokenizer(sample_text)
print(tokenized_output)


ADAPTER_MODEL = "/data/hw/llama_7b_finetuning"

trainer.model.save_pretrained(ADAPTER_MODEL)

finetune_model = "/data/hw/llama_7b_finetuning"

test = pd.read_csv('./test.csv')
base_model = 'meta-llama/Meta-Llama-3-8B-Instruct'

finetune_model = AutoModelForCausalLM.from_pretrained(finetune_model, device_map={"":0})
tokenizer = AutoTokenizer.from_pretrained(base_model)


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
        "<start_of_turn> You are a professional professor of the Korean language. "
        "Your task is to restore an obfuscated Korean review into a clear, natural, and grammatically correct Korean review while preserving its original meaning. "
        "You must strictly adhere to the following conditions:\n"
        "1. The number of sentences in the obfuscated text and the restored review must be exactly the same.\n"
        "2. Interpret the original intent accurately and transform it into a natural sentence without distorting the meaning.\n"
        "3. Ensure that the output is grammatically correct and reads smoothly.\n"
        "4. Avoid excessive changes in word choice while improving clarity.\n"
        f"Input: {query}\n"
        "<end_of_turn>\n"
        "<start_of_turn>Assistant:\n"
        "Output: "
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
    
    # 'Output:' 이후의 텍스트 추출
    output_start = generated_text.find("Output:")

    if output_start != -1:
        restored_reviews.append(generated_text[output_start + len("Output:"):].strip())
    else:
        restored_reviews.append(generated_text.strip())


submission = pd.read_csv('./sample_submission.csv', encoding='utf-8-sig')
submission['output'] = restored_reviews
submission['output'] = submission['output'].apply(lambda x: x.split("<end_of_turn>")[0])
submission.to_csv('./submission.csv', index=False, encoding='utf-8-sig')