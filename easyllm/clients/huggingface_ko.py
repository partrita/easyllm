import json
import logging
import os
from typing import Any, Dict, List, Optional, Union

from huggingface_hub import HfFolder, InferenceClient
from nanoid import generate

from easyllm.prompt_utils.base import build_prompt, buildBasePrompt
from easyllm.schema.base import ChatMessage, Usage, dump_object
from easyllm.schema.openai import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
    DeltaMessage,
    EmbeddingsObjectResponse,
    EmbeddingsRequest,
    EmbeddingsResponse,
)
from easyllm.utils import setup_logger

logger = setup_logger()

# 기본 매개변수
api_type = "huggingface"
api_key = (
    os.environ.get(
        "HUGGINGFACE_TOKEN",
    )
    or HfFolder.get_token()
)
api_base = os.environ.get("HUGGINGFACE_API_BASE", None) or "https://api-inference.huggingface.co/models"
api_version = os.environ.get("HUGGINGFACE_API_VERSION", None) or "2023-07-29"
prompt_builder = os.environ.get("HUGGINGFACE_PROMPT", None)
stop_sequences = []
seed = 42


def stream_chat_request(client, prompt, stop, gen_kwargs, model):
    """스트리밍 채팅 요청을 위한 유틸리티 함수입니다."""
    id = f"hf-{generate(size=10)}"
    res = client.text_generation(
        prompt,
        stream=True,
        details=True,
        **gen_kwargs,
    )
    yield dump_object(
        ChatCompletionStreamResponse(
            id=id,
            model=model,
            choices=[ChatCompletionResponseStreamChoice(index=0, delta=DeltaMessage(role="assistant"))],
        )
    )
    # 생성된 각 토큰을 반환합니다.
    reason = None
    for _idx, chunk in enumerate(res):
        # 특수 토큰 건너뛰기
        if chunk.token.special:
            continue
        # 중지 시퀀스를 만나면 중지합니다.
        if chunk.token.text in stop:
            break
        # details가 None이 아니고 details의 finish_reason 키가 None이 아닌지 확인합니다.
        if chunk.details is not None and chunk.details.finish_reason is not None:
            # reason을 finish_reason으로 설정합니다.
            reason = chunk.details.finish_reason
        # 생성된 토큰을 반환합니다.
        yield dump_object(
            ChatCompletionStreamResponse(
                id=id,
                model=model,
                choices=[ChatCompletionResponseStreamChoice(index=0, delta=DeltaMessage(content=chunk.token.text))],
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
사용자 지정 프롬프트 빌더를 사용하려면 huggingface.prompt_builder를 메시지 목록을 가져와 문자열을 반환하는 함수로 설정하세요.
easyllm.prompt_utils에서 기존 프롬프트 빌더를 가져와 사용할 수도 있습니다."""
            )
            prompt = buildBasePrompt(request.messages)
        else:
            prompt = build_prompt(request.messages, prompt_builder)

        # 모델이 URL인 경우 직접 사용합니다.
        if request.model:
            url = f"{api_base}/{request.model}"
            logger.debug(f"URL:\n{url}")
        else:
            url = api_base

        # 클라이언트를 만듭니다.
        client = InferenceClient(url, token=api_key)

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

        # 생성 매개변수를 만듭니다.
        gen_kwargs = {
            "do_sample": True,
            "return_full_text": False,
            "max_new_tokens": request.max_tokens,
            "top_p": float(request.top_p),
            "temperature": float(request.temperature),
            "stop_sequences": stop,
            "repetition_penalty": request.frequency_penalty,
            "top_k": request.top_k,
            "seed": seed,
        }
        if request.top_p == 0:
            gen_kwargs.pop("top_p")
        if request.top_p == 1:
            request.top_p = 0.9999999
        if request.temperature == 0:
            gen_kwargs.pop("temperature")
            gen_kwargs["do_sample"] = False
        logger.debug(f"생성 매개변수:\n{gen_kwargs}")

        if request.stream:
            return stream_chat_request(client, prompt, stop, gen_kwargs, request.model)
        else:
            choices = []
            generated_tokens = 0
            for _i in range(request.n):
                res = client.text_generation(
                    prompt,
                    details=True,
                    **gen_kwargs,
                )
                parsed = ChatCompletionResponseChoice(
                    index=_i,
                    message=ChatMessage(role="assistant", content=res.generated_text),
                    finish_reason=res.details.finish_reason.value,
                )
                generated_tokens += res.details.generated_tokens
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


def stream_completion_request(client, prompt, stop, gen_kwargs, model):
    """완성 채팅 요청을 위한 유틸리티 함수입니다."""
    id = f"hf-{generate(size=10)}"
    res = client.text_generation(
        prompt,
        stream=True,
        details=True,
        **gen_kwargs,
    )
    # 생성된 각 토큰을 반환합니다.
    for _idx, chunk in enumerate(res):
        # 특수 토큰 건너뛰기
        if chunk.token.special:
            continue
        # 중지 시퀀스를 만나면 중지합니다.
        if chunk.token.text in stop:
            break
        # 생성된 토큰을 반환합니다.
        yield dump_object(
            CompletionStreamResponse(
                id=id,
                model=model,
                choices=[CompletionResponseStreamChoice(index=0, text=chunk.token.text, logprobs=chunk.token.logprob)],
            )
        )


class Completion:
    @staticmethod
    def create(
        prompt: Union[str, List[Any]],
        model: Optional[str] = None,
        suffix: Optional[str] = None,
        temperature: float = 0.9,
        top_p: float = 0.6,
        top_k: Optional[int] = 10,
        n: int = 1,
        max_tokens: int = 1024,
        stop: Optional[List[str]] = None,
        stream: bool = False,
        frequency_penalty: Optional[float] = 1.0,
        logprobs: bool = False,
        echo: bool = False,
        debug: bool = False,
    ) -> Dict[str, Any]:
        """
        제공된 프롬프트 및 매개변수에 대한 새 완성을 만듭니다.

        Args:
            prompt (`Union[str, List[Any]]`) 완성에 사용할 텍스트입니다. `prompt_builder`가 설정된 경우
                프롬프트는 `prompt_builder`로 포맷됩니다.
            model (`str`, *optional*, defaults to None) 완성에 사용할 모델입니다. 제공되지 않으면
                기본 URL로 설정됩니다.
            suffix (`str`, *optional*, defaults to None) 정의된 경우 이 접미사를 프롬프트에 추가합니다.
            temperature (`float`, defaults to 0.9): 완성에 사용할 온도입니다.
            top_p (`float`, defaults to 0.6): 완성에 사용할 top_p입니다.
            top_k (`int`, *optional*, defaults to 10): 완성에 사용할 top_k입니다.
            n (`int`, defaults to 1): 생성할 완성 수입니다.
            max_tokens (`int`, defaults to 1024): 생성할 최대 토큰 수입니다.
            stop (`List[str]`, *optional*, defaults to None): 완성에 사용할 중지 시퀀스입니다.
            stream (`bool`, defaults to False): 완성을 스트리밍할지 여부입니다.
            frequency_penalty (`float`, *optional*, defaults to 1.0): 완성에 사용할 빈도 페널티입니다.
            logprobs (`bool`, defaults to False) 로그 확률을 반환할지 여부입니다.
            echo (`bool`, defaults to False) 프롬프트를 에코할지 여부입니다.
            debug (`bool`, defaults to False): 디버그 로깅을 활성화할지 여부입니다.

        Tip: 프롬프트 빌더
            모델에 항상 프롬프트 빌더를 사용해야 합니다.
        """
        if debug:
            logger.setLevel(logging.DEBUG)

        request = CompletionRequest(
            model=model,
            prompt=prompt,
            suffix=suffix,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            n=n,
            max_tokens=max_tokens,
            stop=stop,
            stream=stream,
            frequency_penalty=frequency_penalty,
            logprobs=logprobs,
            echo=echo,
        )

        # 접미사가 있는 경우 포함합니다.
        if request.suffix is not None:
            request.prompt = request.prompt + request.suffix

        if prompt_builder is None:
            logging.warn(
                f"""huggingface.prompt_builder가 설정되지 않았습니다.
입력을 프롬프트 빌더로 사용합니다. 모델로 전송될 프롬프트는 다음과 같습니다.
----------------------------------------
{request.prompt}.
----------------------------------------
사용자 지정 프롬프트 빌더를 사용하려면 huggingface.prompt_builder를 메시지 목록을 가져와 문자열을 반환하는 함수로 설정하세요.
easyllm.prompt_utils에서 기존 프롬프트 빌더를 가져와 사용할 수도 있습니다."""
            )
            prompt = request.prompt
        else:
            prompt = build_prompt(request.prompt, prompt_builder)
        logger.debug(f"모델로 전송될 프롬프트:\n{prompt}")

        # 모델이 URL인 경우 직접 사용합니다.
        if request.model:
            url = f"{api_base}/{request.model}"
            logger.debug(f"URL:\n{url}")
        else:
            url = api_base

        # 클라이언트를 만듭니다.
        client = InferenceClient(url, token=api_key)

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

        # 생성 매개변수를 만듭니다.
        gen_kwargs = {
            "do_sample": True,
            "return_full_text": True if request.echo else False,
            "max_new_tokens": request.max_tokens,
            "top_p": float(request.top_p),
            "temperature": float(request.temperature),
            "stop_sequences": stop,
            "repetition_penalty": request.frequency_penalty,
            "top_k": request.top_k,
            "seed": seed,
        }
        if request.top_p == 0:
            gen_kwargs.pop("top_p")
        if request.top_p == 1:
            request.top_p = 0.9999999
        if request.temperature == 0:
            gen_kwargs.pop("temperature")
            gen_kwargs["do_sample"] = False
        logger.debug(f"생성 매개변수:\n{gen_kwargs}")

        if request.stream:
            return stream_completion_request(client, prompt, stop, gen_kwargs, request.model)
        else:
            choices = []
            generated_tokens = 0
            for _i in range(request.n):
                res = client.text_generation(
                    prompt,
                    details=True,
                    **gen_kwargs,
                )
                parsed = CompletionResponseChoice(
                    index=_i,
                    text=res.generated_text,
                    finish_reason=res.details.finish_reason.value,
                )
                if request.logprobs:
                    parsed.logprobs = res.details.tokens

                generated_tokens += res.details.generated_tokens
                choices.append(parsed)
                logger.debug(f"인덱스 {_i}의 응답:\n{parsed}")
            # 사용량 세부 정보를 계산합니다.
            # TODO: 세부 정보가 수정되면 수정합니다.
            prompt_tokens = int(len(prompt) / 4)
            total_tokens = prompt_tokens + generated_tokens

            return dump_object(
                CompletionResponse(
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


class Embedding:
    @staticmethod
    def create(
        input: Union[str, List[Any]],
        model: Optional[str] = None,
        debug: bool = False,
    ) -> Dict[str, Any]:
        """
        제공된 프롬프트 및 매개변수에 대한 새 임베딩을 만듭니다.

        Args:
            input (`Union[str, List[Any]]`) 임베딩할 문서입니다.
            model (`str`, *optional*, defaults to None) 완성에 사용할 모델입니다. 제공되지 않으면
                기본 URL로 설정됩니다.
            debug (`bool`, defaults to False): 디버그 로깅을 활성화할지 여부입니다.

        Tip: 프롬프트 빌더
            모델에 항상 프롬프트 빌더를 사용해야 합니다.
        """
        if debug:
            logger.setLevel(logging.DEBUG)

        request = EmbeddingsRequest(model=model, input=input)

        # 모델이 URL인 경우 직접 사용합니다.
        if request.model:
            if api_base.endswith("/models"):
                url = f"{api_base.replace('/models', '/pipeline/feature-extraction')}/{request.model}"
            else:
                url = f"{api_base}/{request.model}"
            logger.debug(f"URL:\n{url}")
        else:
            url = api_base

        # 클라이언트를 만듭니다.
        client = InferenceClient(url, token=api_key)

        # 클라이언트는 현재 일괄 요청을 지원하지 않으므로 순차적으로 실행합니다.
        emb = []
        res = client.post(json={"inputs": request.input, "model": request.model, "task": "feature-extraction"})
        parsed_res = json.loads(res.decode())
        if isinstance(request.input, list):
            for idx, i in enumerate(parsed_res):
                emb.append(EmbeddingsObjectResponse(index=idx, embedding=i))
        else:
            emb.append(EmbeddingsObjectResponse(index=0, embedding=parsed_res))

        if isinstance(res, list):
            # TODO: 토큰만 근사화합니다.
            tokens = [int(len(i) / 4) for i in request.input]
        else:
            tokens = int(len(request.input) / 4)

        return dump_object(
            EmbeddingsResponse(
                model=request.model,
                data=emb,
                usage=Usage(prompt_tokens=tokens, total_tokens=tokens),
            )
        )

    @classmethod
    async def acreate(cls, *args, **kwargs):
        """
        제공된 메시지 및 매개변수에 대한 새 채팅 완성을 만듭니다.
        """
        raise NotImplementedError("ChatCompletion.acreate가 구현되지 않았습니다.")
