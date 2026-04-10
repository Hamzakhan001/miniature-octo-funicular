from __future__ import annotations
from dataclasses import dataclass
from openai import OpenAI
from config.settings import get_settings

settings = get_settings()

@dataclass
class LLMResponse:
    answer: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0

class LLMClient:
    def __init__(self):
        self._settings = get_settings()
        self._client = OpenAI(api_key=self._settings.openai_api_key)

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> LLMResponse:
        response = self._client.chat.completions.create(
            model = self._settings.openai_chat_model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        choices = response.choices[0]
        return LLMResponse(
            answer=choices.message.content,
            model=response.model,
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
        )