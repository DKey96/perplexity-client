from enum import StrEnum

import requests

BASE_URL = "https://api.perplexity.ai"
COMPLETION_URL = "/chat/completions"


class PerplexityModels(StrEnum):
    PPLX_7B_CHAT = "pplx-7b-chat"
    PPLX_70B_CHAT = "pplx-70b-chat"
    PPLX_7B_ONLINE = "pplx-7b-online"
    PPLX_70B_ONLINE = "pplx-70b-online"
    LLAMA_2_70B_CHAT = "llama-2-70b-chat"
    CODELLAMA_34B_INSTRUCT = "codellama-34b-instruct"
    MISTRAL_7B_INSTRUCT = "mistral-7b-instruct"
    MIXTRAL_8X7B_INSTRUCT = "mixtral-8x7b-instruct"


class PerplexityClient:
    def __init__(self, api_key: str, base_url: str = BASE_URL) -> None:
        self.api_key = api_key
        self.base_url = base_url

    def chat_completion(
            self,
            messages: list[dict[str, str]],
            model: PerplexityModels = PerplexityModels.MISTRAL_7B_INSTRUCT,
            **kwargs
    ):
        payload = {
            "model": model.value,
            "messages": messages,
        }

        if kwargs.get("presence_penalty") and kwargs.get("frequency_penalty"):
            raise ValueError("You may use only one, frequency_penalty or presence_penalty. Not both.")

        match kwargs:
            case kwargs.get("max_tokens", None):
                payload = payload | {"max_tokens": kwargs["max_tokens"]}
            case kwargs.get("temperature", None):
                temperature = kwargs["temperature"]
                if 2 < temperature < 0:
                    raise ValueError("Temperature must be between 0 and 2 included.")
                payload = payload | {"temperature": temperature}
            case kwargs.get("top_p", None):
                top_k = kwargs["top_p"]
                if 1 < top_k < 0:
                    raise ValueError("top_p must be between 0 and 1 included.")
                payload = payload | {"top_k": top_k}
            case kwargs.get("top_k", None):
                top_k = kwargs["top_k"]
                if 2048 < top_k < 0:
                    raise ValueError("top_k must be between 0 and 2048 included.")
                payload = payload | {"top_k": top_k}
            case kwargs.get("presence_penalty", None):
                presence_penalty = kwargs["presence_penalty"]
                if 2.0 < presence_penalty < -2.0:
                    raise ValueError("presence_penalty must be between -2.0 and 2.0 included.")
                payload = payload | {"presence_penalty": presence_penalty}
            case kwargs.get("frequency_penalty", None):
                top_k = kwargs["frequency_penalty"]
                payload = payload | {"frequency_penalty": top_k}

        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {self.api_key}"
        }

        url = self.base_url + COMPLETION_URL
        response = requests.post(url, json=payload, headers=headers)

        return response.json()
