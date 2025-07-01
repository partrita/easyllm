# 프롬프트 유틸리티

`prompt_utils` 모듈에는 메시지 사전을 `ChatCompletion` 클라이언트와 함께 사용할 수 있는 프롬프트로 변환하는 데 도움이 되는 함수가 포함되어 있습니다.

지원되는 프롬프트 형식:

- [프롬프트 유틸리티](#프롬프트-유틸리티)
  - [클라이언트용 프롬프트 빌더 설정](#클라이언트용-프롬프트-빌더-설정)
  - [Llama 2 채팅 빌더](#Llama-2-채팅-빌더)
  - [Vicuna 채팅 빌더](#Vicuna-채팅-빌더)
  - [Hugging Face ChatML 빌더](#Hugging-Face-ChatML-빌더)
    - [StarChat](#StarChat)
    - [Falcon](#Falcon)
  - [WizardLM 채팅 빌더](#WizardLM-채팅-빌더)
  - [StableBeluga2 채팅 빌더](#StableBeluga2-채팅-빌더)
  - [Open Assistant 채팅 빌더](#Open-Assistant-채팅-빌더)
  - [Anthropic Claude 채팅 빌더](#Anthropic-Claude-채팅-빌더)

프롬프트 유틸리티는 모델 이름을 프롬프트 빌더 함수에 매핑하는 매핑 사전 `PROMPT_MAPPING`도 내보냅니다. 이를 사용하여 환경 변수를 통해 올바른 프롬프트 빌더 함수를 선택할 수 있습니다.

```python
PROMPT_MAPPING = {
    "chatml_falcon": build_chatml_falcon_prompt,
    "chatml_starchat": build_chatml_starchat_prompt,
    "llama2": build_llama2_prompt,
    "open_assistant": build_open_assistant_prompt,
    "stablebeluga": build_stablebeluga_prompt,
    "vicuna": build_vicuna_prompt,
    "wizardlm": build_wizardlm_prompt,
}
```

## 클라이언트용 프롬프트 빌더 설정

```python
from easyllm.clients import huggingface

huggingface.prompt_builder = "llama2" # vicuna, chatml_falcon, chatml_starchat, wizardlm, stablebeluga, open_assistant
```

## Llama 2 채팅 빌더

채팅 대화를 위한 Llama 2 채팅 프롬프트를 만듭니다. [Llama 2 프롬프트 방법](https://huggingface.co/blog/llama2#how-to-prompt-llama-2)에 대한 Hugging Face 블로그에서 자세히 알아보세요. 지원되지 않는 `role`이 있는 `Message`가 전달되면 오류가 발생합니다.

예제 모델:

* [meta-llama/Llama-2-70b-chat-hf](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf)

```python
from easyllm.prompt_utils import build_llama2_prompt

messages=[
    {"role": "system", "content": "당신은 도움이 되는 조수입니다."},
    {"role": "user", "content": "해적 블랙비어드 스타일로 비동기 프로그래밍을 설명해주세요."},
]
prompt = build_llama2_prompt(messages)
```


## Vicuna 채팅 빌더

채팅 대화를 위한 Vicuna 프롬프트를 만듭니다. 지원되지 않는 `role`이 있는 `Message`가 전달되면 오류가 발생합니다. [참조](https://github.com/lm-sys/FastChat/blob/main/docs/vicuna_weights_version.md#prompt-template)

예제 모델:

* [ehartford/WizardLM-13B-V1.0-Uncensored](https://huggingface.co/ehartford/WizardLM-13B-V1.0-Uncensored)


```python
from easyllm.prompt_utils import build_vicuna_prompt

messages=[
    {"role": "system", "content": "당신은 도움이 되는 조수입니다."},
    {"role": "user", "content": "해적 블랙비어드 스타일로 비동기 프로그래밍을 설명해주세요."},
]
prompt = build_vicuna_prompt(messages)
```

## Hugging Face ChatML 빌더

채팅 대화를 위한 Hugging Face ChatML 프롬프트를 만듭니다. Hugging Face ChatML에는 StarChat 또는 Falcon과 같은 다양한 예제 모델에 대한 다양한 프롬프트가 있습니다. 지원되지 않는 `role`이 있는 `Message`가 전달되면 오류가 발생합니다. [참조](https://huggingface.co/HuggingFaceH4/starchat-beta)

예제 모델:
* [HuggingFaceH4/starchat-beta](https://huggingface.co/HuggingFaceH4/starchat-beta)

### StarChat

```python
from easyllm.prompt_utils import build_chatml_starchat_prompt

messages=[
    {"role": "system", "content": "당신은 도움이 되는 조수입니다."},
    {"role": "user", "content": "해적 블랙비어드 스타일로 비동기 프로그래밍을 설명해주세요."},
]
prompt = build_chatml_starchat_prompt(messages)
```

### Falcon

```python
from easyllm.prompt_utils import build_chatml_falcon_prompt

messages=[
    {"role": "system", "content": "당신은 도움이 되는 조수입니다."},
    {"role": "user", "content": "해적 블랙비어드 스타일로 비동기 프로그래밍을 설명해주세요."},
]
prompt = build_chatml_falcon_prompt(messages)
```

## WizardLM 채팅 빌더

채팅 대화를 위한 WizardLM 프롬프트를 만듭니다. 지원되지 않는 `role`이 있는 `Message`가 전달되면 오류가 발생합니다. [참조](https://github.com/nlpxucan/WizardLM/blob/main/WizardLM/src/infer_wizardlm13b.py#L79)

예제 모델:

* [WizardLM/WizardLM-13B-V1.2](https://huggingface.co/WizardLM/WizardLM-13B-V1.2)

```python
from easyllm.prompt_utils import build_wizardlm_prompt

messages=[
    {"role": "system", "content": "당신은 도움이 되는 조수입니다."},
    {"role": "user", "content": "해적 블랙비어드 스타일로 비동기 프로그래밍을 설명해주세요."},
]
prompt = build_wizardlm_prompt(messages)
```

## StableBeluga2 채팅 빌더

채팅 대화를 위한 StableBeluga2 프롬프트를 만듭니다. 지원되지 않는 `role`이 있는 `Message`가 전달되면 오류가 발생합니다. [참조](https://huggingface.co/stabilityai/StableBeluga2)

```python
from easyllm.prompt_utils import build_stablebeluga_prompt

messages=[
    {"role": "system", "content": "당신은 도움이 되는 조수입니다."},
    {"role": "user", "content": "해적 블랙비어드 스타일로 비동기 프로그래밍을 설명해주세요."},
]
prompt = build_stablebeluga_prompt(messages)
```

## Open Assistant 채팅 빌더

Open Assistant ChatML 템플릿을 만듭니다. `<|prompter|>`, `</s>`, `<|system|>` 및 `<|assistant|>` 토큰을 사용합니다. 지원되지 않는 `role`이 있는 `Message`가 전달되면 오류가 발생합니다. [참조](https://huggingface.co/OpenAssistant/llama2-13b-orca-8k-33192)

예제 모델:

* [OpenAssistant/llama2-13b-orca-8k-3319](https://huggingface.co/OpenAssistant/llama2-13b-orca-8k-33192)

```python
from easyllm.prompt_utils import build_open_assistant_prompt

messages=[
    {"role": "system", "content": "당신은 도움이 되는 조수입니다."},
    {"role": "user", "content": "해적 블랙비어드 스타일로 비동기 프로그래밍을 설명해주세요."},
]
prompt = build_open_assistant_prompt(messages)
```

## Anthropic Claude 채팅 빌더

Anthropic Claude 템플릿을 만듭니다. `\n\nHuman:`, `\n\nAssistant:`를 사용합니다. 지원되지 않는 `role`이 있는 `Message`가 전달되면 오류가 발생합니다. [참조](https://docs.anthropic.com/claude/docs/introduction-to-prompt-design)

예제 모델:

* [Bedrock](https://aws.amazon.com/bedrock/claude/)

```python
from easyllm.prompt_utils import build_anthropic_prompt

messages=[
    {"role": "system", "content": "당신은 도움이 되는 조수입니다."},
    {"role": "user", "content": "해적 블랙비어드 스타일로 비동기 프로그래밍을 설명해주세요."},
]
prompt = build_anthropic_prompt(messages)
```
