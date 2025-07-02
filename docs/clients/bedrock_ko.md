# Amazon Bedrock

EasyLLM은 Amazon Bedrock 모델과 상호 작용하기 위한 클라이언트를 제공합니다.

- `bedrock.ChatCompletion` - OpenAI ChatCompletion API와 호환되는 Bedrock 모델과 상호 작용하기 위한 클라이언트입니다.
- `bedrock.Completion` - OpenAI Completion API와 호환되는 Bedrock 모델과 상호 작용하기 위한 클라이언트입니다.
- `bedrock.Embedding` - OpenAI Embedding API와 호환되는 Bedrock 모델과 상호 작용하기 위한 클라이언트입니다.

## `bedrock.ChatCompletion`

`bedrock.ChatCompletion` 클라이언트는 OpenAI ChatCompletion API와 호환되는 텍스트 생성 추론에서 실행되는 Bedrock 모델과 상호 작용하는 데 사용됩니다. [예제](../examples/bedrock-chat-completion-api)를 확인하세요.


```python
import os
# 프롬프트 빌더용 환경 변수 설정
os.environ["BEDROCK_PROMPT"] = "anthropic" # vicuna, wizardlm, stablebeluga, open_assistant
os.environ["AWS_REGION"] = "us-east-1"  # 사용자 지역으로 변경
# os.environ["AWS_ACCESS_KEY_ID"] = "XXX" # boto3 세션을 사용하지 않는 경우 필요
# os.environ["AWS_SECRET_ACCESS_KEY"] = "XXX" # boto3 세션을 사용하지 않는 경우 필요

from easyllm.clients import bedrock

response = bedrock.ChatCompletion.create(
    model="anthropic.claude-v2",
    messages=[
        {"role": "user", "content": "2 + 2는 무엇인가요?"},
    ],
      temperature=0.9,
      top_p=0.6,
      max_tokens=1024,
      debug=False,
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
* `debug` - 디버그 로깅을 활성화할지 여부입니다. 기본값은 False입니다.


### 프롬프트 빌드

기본적으로 `bedrock` 클라이언트는 `BEDROCK_PROMPT` 환경 변수를 읽고 값을 `PROMPT_MAPPING` 사전에 매핑하려고 시도합니다. 이것이 설정되지 않으면 기본 프롬프트 빌더를 사용합니다.
수동으로 설정할 수도 있습니다.

자세한 내용은 [프롬프트 유틸리티](../prompt_utils)를 확인하세요.


프롬프트 빌더를 수동으로 설정:

```python
from easyllm.clients import bedrock

bedrock.prompt_builder = "anthropic"

res = bedrock.ChatCompletion.create(...)
```

환경 변수 사용:

```python
# 다른 곳에서 발생할 수 있음
import os
os.environ["BEDROCK_PROMPT"] = "anthropic"

from easyllm.clients import bedrock
```
