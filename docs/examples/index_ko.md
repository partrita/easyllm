# 예제

easyllm 라이브러리를 시작하는 데 도움이 되는 몇 가지 예제는 다음과 같습니다.

## Hugging Face

| 예제                                                                 | 설명                                                                            |
| ----------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| [자세한 ChatCompletion 예제](chat-completion-api)                  | ChatCompletion API를 사용하여 모델과 대화형 채팅을 하는 방법을 보여줍니다.  |
| [자세한 Completion 예제](text-completion-api)                      | TextCompletion API를 사용하여 모델로 텍스트를 생성합니다.                           |
| [임베딩 생성](get-embeddings)                                     | 모델을 사용하여 텍스트를 벡터 표현으로 임베딩합니다.                               |
| [채팅 요청 스트리밍 방법 예제](stream-chat-completions)          | 여러 채팅 요청을 스트리밍하여 모델과 효율적으로 채팅하는 방법을 보여줍니다.      |
| [텍스트 요청 스트리밍 방법 예제](stream-text-completions)          | 여러 텍스트 완성 요청을 스트리밍하는 방법을 보여줍니다.                                 |
| [Hugging Face Inference Endpoints 예제](inference-endpoints-example) | Inference Endpoints 또는 localhost와 같은 사용자 지정 엔드포인트를 사용하는 방법에 대한 예제입니다.         |
| [Llama 2를 사용한 검색 증강 생성](llama2-rag-example)      | 컨텍스트 내 검색 증강에 Llama 2 70B를 사용하는 방법에 대한 예제입니다.                 |
| [Llama 2 70B 에이전트/도구 사용 예제](llama2-agent-example)             | Llama 2 70B를 사용하여 도구와 상호 작용하고 에이전트로 사용할 수 있는 방법에 대한 예제입니다. |

이 예제는 라이브러리의 주요 기능인 채팅, 텍스트 완성 및 임베딩을 다룹니다. 어떤 식으로든 색인 페이지를 수정하거나 확장하고 싶다면 알려주십시오.

## Amazon SageMaker

| 예제                                                          | 설명                                                                           |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| [자세한 ChatCompletion 예제](sagemaker-chat-completion-api) | ChatCompletion API를 사용하여 모델과 대화형 채팅을 하는 방법을 보여줍니다. |
| [자세한 Completion 예제](sagemaker-text-completion-api)     | TextCompletion API를 사용하여 모델로 텍스트를 생성합니다.                          |
| [임베딩 생성](sagemaker-get-embeddings)                    | 모델을 사용하여 텍스트를 벡터 표현으로 임베딩합니다.                              |

## Amazon Bedrock

| 예제                                                                | 설명                                                                           |
| ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| [자세한 ChatCompletion 예제](bedrock-chat-completion-api)         | ChatCompletion API를 사용하여 모델과 대화형 채팅을 하는 방법을 보여줍니다. |
| [채팅 요청 스트리밍 방법 예제](bedrock-stream-chat-completions) | 여러 채팅 요청을 스트리밍하여 모델과 효율적으로 채팅하는 방법을 보여줍니다.     |
