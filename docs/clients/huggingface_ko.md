# Hugging Face

EasyLLM은 HuggingFace 모델과 상호 작용하기 위한 클라이언트를 제공합니다. 이 클라이언트는 [HuggingFace Inference API](https://huggingface.co/docs/api-inference/index), [Hugging Face Inference Endpoints](https://huggingface.co/docs/inference-endpoints/index) 또는 [Text Generation Inference](https://github.com/huggingface/text-generation-inference) 또는 호환되는 API 엔드포인트를 실행하는 모든 웹 서비스와 호환됩니다.

- `huggingface.ChatCompletion` - OpenAI ChatCompletion API와 호환되는 HuggingFace 모델과 상호 작용하기 위한 클라이언트입니다.
- `huggingface.Completion` - OpenAI Completion API와 호환되는 HuggingFace 모델과 상호 작용하기 위한 클라이언트입니다.
- `huggingface.Embedding` - OpenAI Embedding API와 호환되는 HuggingFace 모델과 상호 작용하기 위한 클라이언트입니다.

## `huggingface.ChatCompletion`

`huggingface.ChatCompletion` 클라이언트는 OpenAI ChatCompletion API와 호환되는 텍스트 생성 추론에서 실행되는 HuggingFace 모델과 상호 작용하는 데 사용됩니다. 자세한 내용은 [예제](../examples/chat-completion-api)를 확인하고 요청을 스트리밍하는 방법의 예는 [완성 스트리밍 방법](../examples/stream-chat-completion-api)을 참조하세요.


```python
from easyllm.clients import huggingface

# 이 모듈은 환경 변수 HUGGINGFACE_TOKEN 또는 HuggingFace CLI 구성 파일에서 HuggingFace API 키를 자동으로 로드합니다.
# huggingface.api_key="hf_xxx"
huggingface.prompt_builder = "llama2"

response = huggingface.ChatCompletion.create(
    model="meta-llama/Llama-2-70b-chat-hf",
    messages=[
        {"role": "system", "content": "\n당신은 도움이 되고 정중하며 정직한 조수입니다."},
        {"role": "user", "content": "똑똑."},
    ],
    temperature=0.9,
    top_p=0.6,
    max_tokens=1024,
)
```


지원되는 매개변수는 다음과 같습니다.

* `model` - 완성에 사용할 모델입니다. 제공되지 않으면 기본 URL로 기본 설정됩니다.
* `messages` - 완성에 사용할 `List[ChatMessage]`입니다.
* `temperature` - 완성에 사용할 온도입니다. 기본값은 0.9입니다.
* `top_p` - 완성에 사용할 top_p입니다. 기본값은 0.6입니다.
* `top_k` - 완성에 사용할 top_k입니다. 기본값은 10입니다.
* `n` - 생성할 완성 수입니다. 기본값은 1입니다.
* `max_tokens` - 생성할 최대 토큰 수입니다. 기본값은 1024입니다.
* `stop` - 완성에 사용할 중지 시퀀스입니다. 기본값은 None입니다.
* `stream` - 완성을 스트리밍할지 여부입니다. 기본값은 False입니다.
* `frequency_penalty` - 완성에 사용할 빈도 페널티입니다. 기본값은 1.0입니다.
* `debug` - 디버그 로깅을 활성화할지 여부입니다. 기본값은 False입니다.

## `huggingface.Completion`

`huggingface.Completion` 클라이언트는 OpenAI Completion API와 호환되는 텍스트 생성 추론에서 실행되는 HuggingFace 모델과 상호 작용하는 데 사용됩니다. 자세한 내용은 [예제](../examples/text-completion-api)를 확인하고 요청을 스트리밍하는 방법의 예는 [완성 스트리밍 방법](../examples/stream-text-completion-api)을 참조하세요.


```python
from easyllm.clients import huggingface

# 이 모듈은 환경 변수 HUGGINGFACE_TOKEN 또는 HuggingFace CLI 구성 파일에서 HuggingFace API 키를 자동으로 로드합니다.
# huggingface.api_key="hf_xxx"
hubbingface.prompt_builder = "llama2"

response = huggingface.Completion.create(
    model="meta-llama/Llama-2-70b-chat-hf",
    prompt="삶의 의미는 무엇인가요?",
    temperature=0.9,
    top_p=0.6,
    max_tokens=1024,
)
```


지원되는 매개변수는 다음과 같습니다.

* `model` - 완성에 사용할 모델입니다. 제공되지 않으면 기본 URL로 기본 설정됩니다.
* `prompt` - 완성에 사용할 텍스트입니다. `prompt_builder`가 설정된 경우 프롬프트는 `prompt_builder`로 포맷됩니다.
* `temperature` - 완성에 사용할 온도입니다. 기본값은 0.9입니다.
* `top_p` - 완성에 사용할 top_p입니다. 기본값은 0.6입니다.
* `top_k` - 완성에 사용할 top_k입니다. 기본값은 10입니다.
* `n` - 생성할 완성 수입니다. 기본값은 1입니다.
* `max_tokens` - 생성할 최대 토큰 수입니다. 기본값은 1024입니다.
* `stop` - 완성에 사용할 중지 시퀀스입니다. 기본값은 None입니다.
* `stream` - 완성을 스트리밍할지 여부입니다. 기본값은 False입니다.
* `frequency_penalty` - 완성에 사용할 빈도 페널티입니다. 기본값은 1.0입니다.
* `debug` - 디버그 로깅을 활성화할지 여부입니다. 기본값은 False입니다.
* `echo` - 프롬프트를 에코할지 여부입니다. 기본값은 False입니다.
* `logprobs` - 로그 확률을 반환할지 여부입니다. 기본값은 None입니다.


## `huggingface.Embedding`

`huggingface.Embedding` 클라이언트는 OpenAI Embedding API와 호환되는 API로 실행되는 HuggingFace 모델과 상호 작용하는 데 사용됩니다. 자세한 내용은 [예제](../examples/get-embeddings)를 확인하세요.

```python
from easyllm.clients import huggingface

# 이 모듈은 환경 변수 HUGGINGFACE_TOKEN 또는 HuggingFace CLI 구성 파일에서 HuggingFace API 키를 자동으로 로드합니다.
# huggingface.api_key="hf_xxx"

embedding = huggingface.Embedding.create(
    model="sentence-transformers/all-MiniLM-L6-v2",
    text="삶의 의미는 무엇인가요?",
)

len(embedding["data"][0]["embedding"])
```

지원되는 매개변수는 다음과 같습니다.

* `model` - 임베딩을 만드는 데 사용할 모델입니다. 제공되지 않으면 기본 URL로 기본 설정됩니다.
* `input` - 임베딩할 `Union[str, List[str]]` 문서입니다.


## 환경 구성

환경 변수를 설정하거나 기본값을 덮어써서 `huggingface` 클라이언트를 구성할 수 있습니다. HF 토큰, URL 및 프롬프트 빌더를 조정하는 방법은 아래를 참조하세요.

### HF 토큰 설정

기본적으로 `huggingface` 클라이언트는 `HUGGINGFACE_TOKEN` 환경 변수를 읽으려고 시도합니다. 이것이 설정되지 않으면 `~/.huggingface` 폴더에서 토큰을 읽으려고 시도합니다. 이것이 설정되지 않으면 토큰을 사용하지 않습니다.

또는 `huggingface.api_key`를 설정하여 토큰을 수동으로 설정할 수 있습니다.


API 키를 수동으로 설정:

```python
from easyllm.clients import huggingface

huggingface.api_key="hf_xxx"

res = huggingface.ChatCompletion.create(...)
```

환경 변수 사용:

```python
# 다른 곳에서 발생할 수 있음
import os
os.environ["HUGGINGFACE_TOKEN"] = "hf_xxx"

from easyllm.clients import huggingface
```


### URL 변경

기본적으로 `huggingface` 클라이언트는 `HUGGINGFACE_API_BASE` 환경 변수를 읽으려고 시도합니다. 이것이 설정되지 않으면 기본 URL `https://api-inference.huggingface.co/models`를 사용합니다. 이는 `https://zj5lt7pmzqzbp0d1.us-east-1.aws.endpoints.huggingface.cloud`와 같은 다른 URL이나 `http://localhost:8000`과 같은 로컬 URL 또는 Hugging Face Inference Endpoint를 사용하려는 경우에 유용합니다.

또는 `huggingface.api_base`를 설정하여 URL을 수동으로 설정할 수 있습니다. 사용자 지정 URL을 설정하는 경우 `model` 매개변수를 비워 두어야 합니다.

API 베이스를 수동으로 설정:

```python
from easyllm.clients import huggingface

huggingface.api_base="https://my-url"


res = huggingface.ChatCompletion.create(...)
```

환경 변수 사용:

```python
# 다른 곳에서 발생할 수 있음
import os
os.environ["HUGGINGFACE_API_BASE"] = "https://my-url"

from easyllm.clients import huggingface
```




### 프롬프트 빌드

기본적으로 `huggingface` 클라이언트는 `HUGGINGFACE_PROMPT` 환경 변수를 읽고 값을 `PROMPT_MAPPING` 사전에 매핑하려고 시도합니다. 이것이 설정되지 않으면 기본 프롬프트 빌더를 사용합니다.
수동으로 설정할 수도 있습니다.

자세한 내용은 [프롬프트 유틸리티](../prompt_utils)를 확인하세요.


프롬프트 빌더를 수동으로 설정:

```python
from easyllm.clients import huggingface

huggingface.prompt_builder = "llama2"

res = huggingface.ChatCompletion.create(...)
```

환경 변수 사용:

```python
# 다른 곳에서 발생할 수 있음
import os
os.environ["HUGGINGFACE_PROMPT"] = "llama2"

from easyllm.clients import huggingface
```
