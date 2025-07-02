# EasyLLM

EasyLLM은 오픈 소스 및 클로즈드 소스 대규모 언어 모델(LLM) 작업을 위한 유용한 도구와 방법을 제공하는 오픈 소스 프로젝트입니다.

EasyLLM은 OpenAI의 Completion API와 호환되는 클라이언트를 구현합니다. 즉, `openai.ChatCompletion`을 예를 들어 `huggingface.ChatCompletion`으로 쉽게 바꿀 수 있습니다.

* [ChatCompletion 클라이언트](./clients)
* [프롬프트 유틸리티](./prompt_utils)
* [예제](./examples)

## 🚀 시작하기

pip를 통해 EasyLLM을 설치합니다:

```bash
pip install easyllm
```

그런 다음 클라이언트를 가져와서 사용하기 시작합니다:

```python

from easyllm.clients import huggingface

# llama2 프롬프트를 빌드하는 헬퍼
huggingface.prompt_builder = "llama2"

response = huggingface.ChatCompletion.create(
    model="meta-llama/Llama-2-70b-chat-hf",
    messages=[
        {"role": "system", "content": "\n당신은 해적처럼 말하는 도움이 되는 조수입니다. 아르!"},
        {"role": "user", "content": "태양이란 무엇인가?"},
    ],
    temperature=0.9,
    top_p=0.6,
    max_tokens=256,
)

print(response)
```
결과는 다음과 같습니다

```bash
{
  "id": "hf-lVC2iTMkFJ",
  "object": "chat.completion",
  "created": 1690661144,
  "model": "meta-llama/Llama-2-70b-chat-hf",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": " 아르르, 태양은 하늘에 떠 있는 커다란 불덩어리라네, 친구! 우리 아름다운 행성에 빛과 따스함을 주는 원천이고, 강력한 힘을 가졌지, 알겠나? 태양이 없다면 우린 어둠 속을 항해하며 길을 잃고 추위에 떨게 될 테니, 태양을 위해 힘차게 \"야르!\"를 외치자고, 친구들! 아르르!"
      },
      "finish_reason": null
    }
  ],
  "usage": {
    "prompt_tokens": 111,
    "completion_tokens": 299,
    "total_tokens": 410
  }
}
```

다른 예제를 확인하세요:

* [자세한 ChatCompletion 예제](examples/chat-completion-api)
* [채팅 요청 스트리밍 방법 예제](examples/stream-chat-completion)
* [텍스트 요청 스트리밍 방법 예제](examples/stream-text-completion)
* [자세한 Completion 예제](examples/text-completion-api)
* [임베딩 생성](examples/get-embeddings)


## 💪🏻 OpenAI에서 HuggingFace로 마이그레이션

OpenAI에서 HuggingFace로 마이그레이션하는 것은 쉽습니다. 가져오기 문과 사용하려는 클라이언트를 변경하고 선택적으로 프롬프트 빌더를 변경하면 됩니다.

```diff
- import openai
+ from easyllm.clients import huggingface
+ huggingface.prompt_builder = "llama2"


- response = openai.ChatCompletion.create(
+ response = huggingface.ChatCompletion.create(
-    model="gpt-3.5-turbo",
+    model="meta-llama/Llama-2-70b-chat-hf",
    messages=[
        {"role": "system", "content": "당신은 도움이 되는 조수입니다."},
        {"role": "user", "content": "똑똑."},
    ],
)
```

클라이언트를 전환할 때 하이퍼파라미터가 여전히 유효한지 확인하세요. 예를 들어 GPT-3의 `temperature`는 `Llama-2`의 `temperature`와 다를 수 있습니다.

## ☑️ 주요 기능

### 🤝 호환되는 클라이언트

- `openai.ChatCompletion`의 OpenAI API 형식과 호환되는 클라이언트 구현.
- 코드 한 줄을 변경하여 `openai.ChatCompletion`과 `huggingface.ChatCompletion`과 같은 다른 LLM 간에 쉽게 전환할 수 있습니다.
- 완성 스트리밍 지원, [완성 스트리밍 방법](examples/stream-chat-completions) 예제 확인.

### ⚙️ 헬퍼 모듈 ⚙️

- `evol_instruct` (작업 진행 중) - 진화 알고리즘을 사용하여 LLM용 지침을 만듭니다.

- `prompt_utils` - OpenAI 메시지와 같은 프롬프트 형식을 Llama 2와 같은 오픈 소스 모델용 프롬프트로 쉽게 변환하는 헬퍼 메서드입니다.

## 📔 인용 및 감사

EasyLLM을 사용하신다면 소셜 미디어나 이메일로 저와 공유해주세요. 정말 듣고 싶습니다!
다음 BibTeX을 사용하여 프로젝트를 인용할 수도 있습니다:

```bash
@software{Philipp_Schmid_EasyLLM_2023,
author = {Philipp Schmid},
license = {Apache-2.0},
month = juj,
title = {EasyLLM: Streamlined Tools for LLMs},
url = {https://github.com/philschmid/easyllm},
year = {2023}
}
```

<!-- ## 코드

코드의 함수 링크:
[`객체 1`][easyllm.utils.fancy_function] -->
