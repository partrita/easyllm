# 클라이언트

EasyLLM의 맥락에서 "클라이언트"는 특정 LLM API(예: OpenAI)와 상호 작용하는 코드를 의미합니다.

현재 지원되는 클라이언트는 다음과 같습니다.

- `ChatCompletion` - ChatCompletion 클라이언트는 OpenAI ChatCompletion API와 호환되는 LLM과 상호 작용하는 데 사용됩니다.
- `Completion` - Completion 클라이언트는 OpenAI Completion API와 호환되는 LLM과 상호 작용하는 데 사용됩니다.
- `Embedding` - Embedding 클라이언트는 OpenAI Embedding API와 호환되는 LLM과 상호 작용하는 데 사용됩니다.

현재 지원되는 클라이언트는 다음과 같습니다.

## Hugging Face

- [huggingface.ChatCompletion](huggingface/#huggingfacechatcompletion) - OpenAI ChatCompletion API와 호환되는 HuggingFace 모델과 상호 작용하기 위한 클라이언트입니다.
- [huggingface.Completion](huggingface/#huggingfacechatcompletion) - OpenAI Completion API와 호환되는 HuggingFace 모델과 상호 작용하기 위한 클라이언트입니다.
- [huggingface.Embedding](huggingface/#huggingfacechatcompletion) - OpenAI Embedding API와 호환되는 HuggingFace 모델과 상호 작용하기 위한 클라이언트입니다.

## Amazon SageMaker

- [sagemaker.ChatCompletion](sagemaker/#sagemakerchatcompletion) - OpenAI ChatCompletion API와 호환되는 Amazon SageMaker 모델과 상호 작용하기 위한 클라이언트입니다.
- [sagemaker.Completion](sagemaker/#sagemakercompletion) - OpenAI Completion API와 호환되는 Amazon SageMaker 모델과 상호 작용하기 위한 클라이언트입니다.
- [sagemaker.Embedding](sagemaker/#sagemakerembedding) - OpenAI Embedding API와 호환되는 Amazon SageMaker 모델과 상호 작용하기 위한 클라이언트입니다.

## Amazon Bedrock

- [bedrock.ChatCompletion](bedrock/#bedrockchatcompletion) - OpenAI ChatCompletion API와 호환되는 Amazon Bedrock 모델과 상호 작용하기 위한 클라이언트입니다.
