from openai import OpenAI
from difftrust.config import config
from difftrust.llm.abstract import LLM
from difftrust.llm.chat import Chat, Msg


class OpenRouterLLM(LLM):
    def __init__(self, model: str, temperature: float = 0.6):
        super().__init__(model, temperature)
        self.model = model
        self.temperature = temperature
        self.failed_msg = None
        try:
            self.client = OpenAI(
                api_key=config()["llm"]["open_router_api_key"],
                base_url="https://openrouter.ai/api/v1"
            )
        except Exception as exception:
            self.failed_msg = f"Something went wrong when charging {model} : {exception}"

    def run(self, chat: Chat) -> Msg:
        if self.failed_msg is not None:
            raise Exception(self.failed_msg)
        messages = [{"role": msg.writer, "content": msg.content} for msg in chat.chatstream]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature
        )
        return Msg("assistant", response.choices[0].message.content)


# LLAMA
llama_3_1_70b_instruct = OpenRouterLLM(model="meta-llama/llama-3.1-70b-instruct")
llama_4_maverick_17b = OpenRouterLLM(model="meta-llama/llama-4-maverick")
llama_3_3_70b_instruct = OpenRouterLLM(model="meta-llama/llama-3.3-70b-instruct")

# DeepSeek
deepseek_v3_0324 = OpenRouterLLM(model="deepseek/deepseek-chat-v3-0324", temperature=0.6)
deepseek_chat = OpenRouterLLM(model="deepseek/deepseek-chat", temperature=0.6)
deepseek_r1 = OpenRouterLLM(model="deepseek/deepseek-r1", temperature=0.6)

# Medium sized
devstral_small = OpenRouterLLM(model="mistralai/devstral-small")
qwen3_14b = OpenRouterLLM(model="qwen/qwen3-14b")
gemma_3_1b_it = OpenRouterLLM(model="google/gemma-3-12b-it")

# Small sized
ministral_8b = OpenRouterLLM(model="mistralai/ministral-8b")
llama_3_1_8b_instruct = OpenRouterLLM(model="meta-llama/llama-3.1-8b-instruct")
qwen3_8b = OpenRouterLLM(model="qwen/qwen3-8b")
