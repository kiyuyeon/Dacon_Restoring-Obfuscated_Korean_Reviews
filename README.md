# Restoring-Obfuscated_Korean_Reviews

---
# 월간데이콘 '난독화된 한글 리뷰 복원 AI 경진대회'에 오신 것을 환영합니다!



해외 숙소 예약 사이트에서는 나쁜 평을 남기면 삭제될 수 있기 때문에

이를 피하면서 한국인들에게만 유용한 정보를 전달하기 위해 한글을 난독화 하는 방법이 등장했습니다.

이러한 방식은 한국인에게는 솔직한 리뷰를 바탕으로 숙소를 선택할 수 있는 장점을 제공하지만, 

숙소 운영자와 방문객들에게 피드백 전달을 어렵게 만들어 서비스의 품질을 향상에 문제가 될 수 있습니다.


이번 월간 데이콘에서는 이러한 난독화된 한글 리뷰를 원래의 명확한 내용의 리뷰로 복원하는 알고리즘을 개발하는 것을 목표로 합니다.

---

이 저장소는 변형된 한국어 텍스트를 원래 형태로 복원하는 작업을 위해 설계된 PyTorch `Dataset` 클래스를 제공합니다. Hugging Face의 `transformers` 라이브러리와 호환되며, Meta의 Llama-3와 같은 언어 모델을 미세 조정하는 데 사용할 수 있습니다.

## 주요 기능
- **텍스트 전처리**: `prompt`와 `response`를 결합하여 모델 입력을 생성합니다.
- **토큰화**: Hugging Face 토크나이저를 사용해 텍스트를 토큰화하고, `input_ids`, `attention_mask`, `labels`를 생성합니다.
- **최대 길이 제어**: `max_length`를 설정하여 입력 시퀀스의 길이를 조절합니다.

## 사용 방법

### 1. **토크나이저 로드**
Hugging Face의 `AutoTokenizer`를 사용해 토크나이저를 로드합니다. 패딩 토큰은 `eos_token`으로 설정됩니다.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", cache_dir="H:/model/")
tokenizer.pad_token = tokenizer.eos_token
```

### 2. **데이터셋 생성**
`CustomDataset` 클래스를 사용해 데이터셋을 생성합니다. 입력 데이터는 Pandas DataFrame 형태여야 하며, `input`과 `output` 열을 포함해야 합니다.

```python
dataset = CustomDataset(train, tokenizer)
```

### 3. **데이터셋 구조**
`CustomDataset`은 다음과 같은 항목을 반환합니다:
- **`input_ids`**: 토큰화된 입력 텍스트.
- **`attention_mask`**: 패딩 토큰을 무시하기 위한 마스크.
- **`labels`**: 모델이 예측해야 할 대상 텍스트 (`response`).

---



## 요구 사항
- Python 3.8+
- PyTorch
- Hugging Face `transformers`
- Pandas
---

