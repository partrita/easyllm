<h1 align="center">EasyLLM - </h1>

<div align="center">
	<a  href="https://pypi.org/project/easyllm" target="_blank">
		<img src="https://img.shields.io/pypi/v/easyllm.svg" />
	</a>
	<a  href="https://pypi.org/project/easyllm" target="_blank">
		<img src="https://img.shields.io/pypi/pyversions/easyllm" />
	</a>
	<a  href="https://github.com/philschmid/easyllm/blob/main/LICENSE" target="_blank">
		<img src="https://img.shields.io/pypi/l/easyllm" />
	</a>
	<a  href="https://github.com/philschmid/easyllm/actions?workflow=Unit Tests" target="_blank">
		<img src="https://github.com/philschmid/easyllm/workflows/Unit Tests/badge.svg" />
	</a>
  <a  href="https://github.com/pypa/hatch" target="_blank">
		<img src="https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg" />
	</a>
</div>


**EasyLLM**은 오픈 소스 및 클로즈드 소스 대규모 언어 모델(LLM) 작업을 위한 **유용한 도구와 방법**을 제공하는 오픈 소스 프로젝트입니다. 즉시 시작하거나 [문서](https://philschmid.github.io/easyllm/)를 확인하세요.

EasyLLM은 **OpenAI의 Completion API와 호환되는 클라이언트**를 구현합니다. 즉, 코드 한 줄을 변경하여 `openai.ChatCompletion`, `openai.Completion`, `openai.Embedding`을 예를 들어 `huggingface.ChatCompletion`, `huggingface.Completion` 또는 `huggingface.Embedding`으로 쉽게 바꿀 수 있습니다.

### 지원되는 클라이언트

* `huggingface` - [HuggingFace](https://huggingface.co/) 모델
  * `huggingface.ChatCompletion` - LLM과 채팅
  * `huggingface.Completion` - LLM으로 텍스트 완성
  * `huggingface.Embedding` - LLM으로 임베딩 생성
* `sagemaker` - Amazon SageMaker에 배포된 오픈 LLM
  * `sagemaker.ChatCompletion` - LLM과 채팅
  * `sagemaker.Completion` - LLM으로 텍스트 완성
  * `sagemaker.Embedding` - LLM으로 임베딩 생성
* `bedrock` - Amazon Bedrock LLM


시작하려면 [예제](./examples)를 확인하세요.

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
* [자세한 ChatCompletion 예제](notebooks/chat-completion-api.ipynb)
* [채팅 요청을 스트리밍하는 방법 예제](notebooks/stream-chat-completions.ipynb)
* [텍스트 요청을 스트리밍하는 방법 예제](notebooks/stream-text-completions.ipynb)
* [자세한 Completion 예제](notebooks/text-completion-api.ipynb)
* [임베딩 생성](notebooks/get-embeddings)

더 자세한 사용법과 예제는 [문서](https://philschmid.github.io/easyllm/)를 참조하세요.

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

- `openai.ChatCompletion`, `openai.Completion`, `openai.Embedding`의 OpenAI API 형식과 호환되는 클라이언트 구현.
- 코드 한 줄을 변경하여 `openai.ChatCompletion`과 `huggingface.ChatCompletion`과 같은 다른 LLM 간에 쉽게 전환할 수 있습니다.
- 완성 스트리밍 지원, [완성 스트리밍 방법](./notebooks/stream-chat-completions.ipynb) 예제 확인.

### ⚙️ 헬퍼 모듈 ⚙️

- `evol_instruct` (작업 진행 중) - 진화 알고리즘을 사용하여 LLM용 지침을 만듭니다.

- `prompt_utils` - OpenAI 메시지와 같은 프롬프트 형식을 Llama 2와 같은 오픈 소스 모델용 프롬프트로 쉽게 변환하는 헬퍼 메서드입니다.

## 🙏 기여

EasyLLM은 오픈 소스 프로젝트이며 모든 종류의 기여를 환영합니다.

이 프로젝트는 개발에 [hatch](https://hatch.pypa.io/latest/)를 사용합니다. 시작하려면 리포지토리를 포크하고 로컬 시스템에 복제하세요.

0. [hatch](https://hatch.pypa.io/latest/install/)가 설치되어 있는지 확인합니다 (pipx는 시스템 전체에서 사용할 수 있도록 하는 데 유용합니다).
1. 프로젝트 디렉터리에서 `hatch env create`를 실행하여 개발용 기본 가상 환경을 만듭니다.
2. `hatch shell`로 가상 환경을 활성화합니다.
3. 개발을 시작하세요! 🤩

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
