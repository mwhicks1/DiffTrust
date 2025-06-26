import openai
from difftrust.config import config
from difftrust.llm.abstract import LLM
from difftrust.llm.chat import Chat, Msg

# Set up API key
openai.api_key = config()["llm"]["openai_api_key"]

_no_temperature = {"o1-2024-12-17", "o3-mini-2025-01-31", "o4-mini"}


class OpenAIModel(LLM):
    def __init__(self, model: str = "gpt-3.5-turbo", temperature: float = 0.6):
        super().__init__(model, temperature)
        self.model = model
        self.failed_msg = None
        try:
            self.client = openai.OpenAI(api_key=config()["llm"]["openai_api_key"])
        except Exception as exception:
            self.failed_msg = f"Something went wrong when charging {model} : {exception}"

    def run(self, chat: Chat) -> Msg:
        if self.failed_msg is not None:
            raise Exception(self.failed_msg)
        messages = [{"role": msg.writer, "content": msg.content} for msg in chat.chatstream]
        if self.model in _no_temperature:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature
            )
        return Msg("assistant", response.choices[0].message.content)


# Example instances
gpt_3_5 = OpenAIModel("gpt-3.5-turbo", temperature=0.6)
gpt_3_5_16k = OpenAIModel("gpt-3.5-turbo-16k", temperature=0.6)
gpt_4 = OpenAIModel("gpt-4", temperature=0.6)
gpt_4_turbo = OpenAIModel("gpt-4-turbo", temperature=0.6)
gpt_4_preview = OpenAIModel("gpt-4-1106-preview", temperature=0.6)
gpt_4o = OpenAIModel(model="gpt-4o", temperature=0.6)
gpt_o1 = OpenAIModel(model="o1-2024-12-17", temperature=0.0)
gpt_o3_mini = OpenAIModel(model="o3-mini-2025-01-31", temperature=0.0)
gpt_o4_mini = OpenAIModel(model="o4-mini", temperature=0.0)
