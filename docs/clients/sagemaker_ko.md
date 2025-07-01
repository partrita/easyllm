# Amazon SageMaker

EasyLLM은 Amazon SageMaker 모델과 상호 작용하기 위한 클라이언트를 제공합니다.

- `sagemaker.ChatCompletion` - OpenAI ChatCompletion API와 호환되는 sagemaker 모델과 상호 작용하기 위한 클라이언트입니다.
- `sagemaker.Completion` - OpenAI Completion API와 호환되는 sagemaker 모델과 상호 작용하기 위한 클라이언트입니다.
- `sagemaker.Embedding` - OpenAI Embedding API와 호환되는 sagemaker 모델과 상호 작용하기 위한 클라이언트입니다.

## `sagemaker.ChatCompletion`

`sagemaker.ChatCompletion` 클라이언트는 OpenAI ChatCompletion API와 호환되는 텍스트 생성 추론에서 실행되는 sagemaker 모델과 상호 작용하는 데 사용됩니다. [예제](../examples/sagemaker-chat-completion-api)를 확인하세요.


```python
import os
from easyllm.clients import sagemaker

# 프롬프트 빌더용 환경 변수 설정
os.environ["HUGGINGFACE_PROMPT"] = "llama2" # vicuna, wizardlm, stablebeluga, open_assistant
os.environ["AWS_REGION"] = "us-east-1"  # 사용자 지역으로 변경
# os.environ["AWS_ACCESS_KEY_ID"] = "XXX" # boto3 세션을 사용하지 않는 경우 필요
# os.environ["AWS_SECRET_ACCESS_KEY"] = "XXX" # boto3 세션을 사용하지 않는 경우 필요


response = sagemaker.ChatCompletion.create(
    model="huggingface-pytorch-tgi-inference-2023-08-08-14-15-52-703",
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

## `sagemaker.Completion`

`sagemaker.Completion` 클라이언트는 OpenAI Completion API와 호환되는 텍스트 생성 추론에서 실행되는 sagemaker 모델과 상호 작용하는 데 사용됩니다. [예제](../examples/sagemaker-text-completion-api)를 확인하세요.


```python
import os
from easyllm.clients import sagemaker

# 프롬프트 빌더용 환경 변수 설정
os.environ["HUGGINGFACE_PROMPT"] = "llama2" # vicuna, wizardlm, stablebeluga, open_assistant
os.environ["AWS_REGION"] = "us-east-1"  # 사용자 지역으로 변경
# os.environ["AWS_ACCESS_KEY_ID"] = "XXX" # boto3 세션을 사용하지 않는 경우 필요
# os.environ["AWS_SECRET_ACCESS_KEY"] = "XXX" # boto3 세션을 사용하지 않는 경우 필요

response = sagemaker.Completion.create(
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


## `sagemaker.Embedding`

`sagemaker.Embedding` 클라이언트는 OpenAI Embedding API와 호환되는 API로 실행되는 sagemaker 모델과 상호 작용하는 데 사용됩니다. 자세한 내용은 [예제](../examples/sagemaker-get-embeddings)를 확인하세요.

```python
import os
# 프롬프트 빌더용 환경 변수 설정
os.environ["HUGGINGFACE_PROMPT"] = "llama2" # vicuna, wizardlm, stablebeluga, open_assistant
os.environ["AWS_REGION"] = "us-east-1"  # 사용자 지역으로 변경
# os.environ["AWS_ACCESS_KEY_ID"] = "XXX" # boto3 세션을 사용하지 않는 경우 필요
# os.environ["AWS_SECRET_ACCESS_KEY"] = "XXX" # boto3 세션을 사용하지 않는 경우 필요

from easyllm.clients import sagemaker

embedding = sagemaker.Embedding.create(
    model="SageMakerModelEmbeddingEndpoint24E49D09-64prhjuiWUtE",
    input="저 차 멋지네요.",
)

len(embedding["data"][0]["embedding"])
```

지원되는 매개변수는 다음과 같습니다.

* `model` - 임베딩을 만드는 데 사용할 모델입니다. 제공되지 않으면 기본 URL로 기본 설정됩니다.
* `input` - 임베딩할 `Union[str, List[str]]` 문서입니다.


## 환경 구성

환경 변수를 설정하거나 기본값을 덮어써서 `sagemaker` 클라이언트를 구성할 수 있습니다. HF 토큰, URL 및 프롬프트 빌더를 조정하는 방법은 아래를 참조하세요.

### 자격 증명 설정

기본적으로 `sagemaker` 클라이언트는 `AWS_ACCESS_KEY_ID` 및 `AWS_SECRET_ACCESS_KEY` 환경 변수를 읽으려고 시도합니다. 이것이 설정되지 않으면 `boto3`를 사용하려고 시도합니다.

또는 `sagemaker.*`를 설정하여 토큰을 수동으로 설정할 수 있습니다.

API 키를 수동으로 설정:

```python
from easyllm.clients import sagemaker

sagemaker.api_aws_access_key="xxx"
sagemaker.api_aws_secret_key="xxx"

res = sagemaker.ChatCompletion.create(...)
```

환경 변수 사용:

```python
# 다른 곳에서 발생할 수 있음
import os
os.environ["AWS_ACCESS_KEY_ID"] = "xxx"
os.environ["AWS_SECRET_ACCESS_KEY"] = "xxx"

from easyllm.clients import sagemaker
```


### 프롬프트 빌드

기본적으로 `sagemaker` 클라이언트는 `sagemaker_PROMPT` 환경 변수를 읽고 값을 `PROMPT_MAPPING` 사전에 매핑하려고 시도합니다. 이것이 설정되지 않으면 기본 프롬프트 빌더를 사용합니다.
수동으로 설정할 수도 있습니다.

자세한 내용은 [프롬프트 유틸리티](../prompt_utils)를 확인하세요.


프롬프트 빌더를 수동으로 설정:

```python
from easyllm.clients import sagemaker

sagemaker.prompt_builder = "llama2"

res = sagemaker.ChatCompletion.create(...)
```

환경 변수 사용:

```python
# 다른 곳에서 발생할 수 있음
import os
os.environ["HUGGINGFACE_PROMPT"] = "llama2"

from easyllm.clients import sagemaker
```
