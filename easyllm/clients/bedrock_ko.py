import json
import logging
import os
from typing import Any, Dict, List, Optional

from nanoid import generate

from easyllm.prompt_utils.base import build_prompt, buildBasePrompt
from easyllm.schema.base import ChatMessage, Usage, dump_object
from easyllm.schema.openai import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    DeltaMessage,
)
from easyllm.utils import setup_logger
from easyllm.utils.aws import get_bedrock_client

logger = setup_logger()

# 기본 매개변수
api_type = "bedrock"
api_aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID", None)
api_aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", None)
api_aws_session_token = os.environ.get("AWS_SESSION_TOKEN", None)

client = get_bedrock_client(
    aws_access_key_id=api_aws_access_key,
    aws_secret_access_key=api_aws_secret_key,
    aws_session_token=api_aws_session_token,
)


SUPPORTED_MODELS = [
    "anthropic.claude-v2",
]
model_version_mapping = {"anthropic.claude-v2": "bedrock-2023-05-31"}

api_version = os.environ.get("BEDROCK_API_VERSION", None) or "bedrock-2023-05-31"
prompt_builder = os.environ.get("BEDROCK_PROMPT", None)
stop_sequences = []


def stream_chat_request(client, body, model):
    """스트리밍 채팅 요청을 위한 유틸리티 함수입니다."""
    id = f"hf-{generate(size=10)}"
    response = client.invoke_model_with_response_stream(
        body=json.dumps(body), modelId=model, accept="application/json", contentType="application/json"
    )
    stream = response.get("body")

    yield dump_object(
        ChatCompletionStreamResponse(
            id=id,
            model=model,
            choices=[ChatCompletionResponseStreamChoice(index=0, delta=DeltaMessage(role="assistant"))],
        )
    )
    # 생성된 각 토큰을 반환합니다.
    reason = None
    for _idx, event in enumerate(stream):
        chunk = event.get("chunk")
        if chunk:
            chunk_obj = json.loads(chunk.get("bytes").decode())
            text = chunk_obj["completion"]
            yield dump_object(
                ChatCompletionStreamResponse(
                    id=id,
                    model=model,
                    choices=[ChatCompletionResponseStreamChoice(index=0, delta=DeltaMessage(content=text))],
                )
            )
    yield dump_object(
        ChatCompletionStreamResponse(
            id=id,
            model=model,
            choices=[ChatCompletionResponseStreamChoice(index=0, finish_reason=reason, delta={})],
        )
    )


class ChatCompletion:
    @staticmethod
    def create(
        messages: List[ChatMessage],
        model: Optional[str] = None,
        temperature: float = 0.9,
        top_p: float = 0.6,
        top_k: Optional[int] = 10,
        n: int = 1,
        max_tokens: int = 1024,
        stop: Optional[List[str]] = None,
        stream: bool = False,
        frequency_penalty: Optional[float] = 1.0,
        debug: bool = False,
    ) -> Dict[str, Any]:
        """
        제공된 메시지 및 매개변수에 대한 새 채팅 완성을 만듭니다.

        Args:
            messages (`List[ChatMessage]`): 완성에 사용할 메시지입니다.
            model (`str`, *optional*, defaults to None): 완성에 사용할 모델입니다. 제공되지 않으면
                기본 URL로 설정됩니다.
            temperature (`float`, defaults to 0.9): 완성에 사용할 온도입니다.
            top_p (`float`, defaults to 0.6): 완성에 사용할 top_p입니다.
            top_k (`int`, *optional*, defaults to 10): 완성에 사용할 top_k입니다.
            n (`int`, defaults to 1): 생성할 완성 수입니다.
            max_tokens (`int`, defaults to 1024): 생성할 최대 토큰 수입니다.
            stop (`List[str]`, *optional*, defaults to None): 완성에 사용할 중지 시퀀스입니다.
            stream (`bool`, defaults to False): 완성을 스트리밍할지 여부입니다.
            frequency_penalty (`float`, *optional*, defaults to 1.0): 완성에 사용할 빈도 페널티입니다.
            debug (`bool`, defaults to False): 디버그 로깅을 활성화할지 여부입니다.

        Tip: 프롬프트 빌더
            모델에 항상 프롬프트 빌더를 사용해야 합니다.
        """
        if debug:
            logger.setLevel(logging.DEBUG)

        # 모델이 model_mapping에 있는지 확인합니다.
        if model not in SUPPORTED_MODELS:
            raise ValueError(f"모델 {model}은(는) 지원되지 않습니다. 지원되는 모델은 다음과 같습니다. {SUPPORTED_MODELS}")

        request = ChatCompletionRequest(
            messages=messages,
            model=model,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            n=n,
            max_tokens=max_tokens,
            stop=stop,
            stream=stream,
            frequency_penalty=frequency_penalty,
        )

        if prompt_builder is None:
            logger.warn(
                f"""huggingface.prompt_builder가 설정되지 않았습니다.
기본 프롬프트 빌더를 사용합니다. 모델로 전송될 프롬프트는 다음과 같습니다.
----------------------------------------
{buildBasePrompt(request.messages)}.
----------------------------------------
사용자 지정 프롬프트 빌더를 사용하려면 bedrock.prompt_builder를 메시지 목록을 가져와 문자열을 반환하는 함수로 설정하세요.
easyllm.prompt_utils에서 기존 프롬프트 빌더를 가져와 사용할 수도 있습니다."""
            )
            prompt = buildBasePrompt(request.messages)
        else:
            prompt = build_prompt(request.messages, prompt_builder)

        # 중지 시퀀스를 만듭니다.
        if isinstance(request.stop, list):
            stop = stop_sequences + request.stop
        elif isinstance(request.stop, str):
            stop = stop_sequences + [request.stop]
        else:
            stop = stop_sequences
        logger.debug(f"중지 시퀀스:\n{stop}")

        # 스트리밍할 수 있는지 확인합니다.
        if request.stream is True and request.n > 1:
            raise ValueError("하나 이상의 완성을 스트리밍할 수 없습니다.")

        # 본문을 구성합니다.
        body = {
            "prompt": prompt,
            "max_tokens_to_sample": request.max_tokens,
            "temperature": request.temperature,
            "top_k": request.top_k,
            "top_p": request.top_p,
            "stop_sequences": stop,
            "anthropic_version": model_version_mapping[model],
        }
        logger.debug(f"생성 본문:\n{body}")

        if request.stream:
            return stream_chat_request(client, body, model)
        else:
            choices = []
            generated_tokens = 0
            for _i in range(request.n):
                response = client.invoke_model(
                    body=json.dumps(body), modelId=model, accept="application/json", contentType="application/json"
                )
                # 응답을 구문 분석합니다.
                res = json.loads(response.get("body").read())

                # 스키마로 변환합니다.
                parsed = ChatCompletionResponseChoice(
                    index=_i,
                    message=ChatMessage(role="assistant", content=res["completion"].strip()),
                    finish_reason=res["stop_reason"],
                )
                generated_tokens += len(res["completion"].strip()) // 4
                choices.append(parsed)
                logger.debug(f"인덱스 {_i}의 응답:\n{parsed}")
            # 사용량 세부 정보를 계산합니다.
            # TODO: 세부 정보가 수정되면 수정합니다.
            prompt_tokens = int(len(prompt) / 4)
            total_tokens = prompt_tokens + generated_tokens

            return dump_object(
                ChatCompletionResponse(
                    model=request.model,
                    choices=choices,
                    usage=Usage(
                        prompt_tokens=prompt_tokens, completion_tokens=generated_tokens, total_tokens=total_tokens
                    ),
                )
            )

    @classmethod
    async def acreate(cls, *args, **kwargs):
        """
        제공된 메시지 및 매개변수에 대한 새 채팅 완성을 만듭니다.
        """
        raise NotImplementedError("ChatCompletion.acreate가 구현되지 않았습니다.")
