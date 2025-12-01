try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    anthropic = None

from difftrust.config import config
from difftrust.llm.abstract import LLM
from difftrust.llm.chat import Chat, Msg


class ClaudeModel(LLM):
    def __init__(self, model: str = "claude-3.5-sonnet", temperature: float = 0.6, max_tokens: int = 16384):
        super().__init__(model, temperature)
        self.model = model
        self.failed_msg = None
        if not ANTHROPIC_AVAILABLE:
            self.failed_msg = "Anthropic package not installed"
        else:
            try:
                self.client = anthropic.Anthropic(api_key=config()["llm"]["claude_api_key"])
            except Exception as exception:
                self.failed_msg = f"Something went wrong when charging {model} : {exception}"
        self.max_tokens = max_tokens

    def run(self, chat: Chat) -> Msg:
        if self.failed_msg is not None:
            raise Exception(self.failed_msg)
        messages = [{"role": msg.writer, "content": msg.content} for msg in chat.chatstream[1:]]
        system_prompt = chat.chatstream[0].content if chat.chatstream else None
        with self.client.messages.stream(
                model=self.model,
                system=system_prompt,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
        ) as stream:
            content = ""
            for text in stream.text_stream:
                content += text
            return Msg("assistant", content)


claude_opus_4 = ClaudeModel(model="claude-opus-4-20250514", temperature=0.6)
claude_sonnet_4 = ClaudeModel(model="claude-sonnet-4-20250514", temperature=0.6)
claude_3_7_sonnet = ClaudeModel(model="claude-3-7-sonnet-20250219", temperature=0.6)
claude_3_5_sonnet = ClaudeModel(model="claude-3-5-sonnet-20241022", temperature=0.6, max_tokens=8192)
claude_3_5_haiku = ClaudeModel(model="claude-3-5-haiku-20241022", temperature=0.6, max_tokens=8192)
claude_3_opus = ClaudeModel(model="claude-3-opus-20240229", temperature=0.6, max_tokens=4096)
# claude_3_sonnet = ClaudeModel(model="claude-3-sonnet-20240229", temperature=0.6, max_tokens=4096) (Deprecated)
claude_3_haiku = ClaudeModel(model="claude-3-haiku-20240307", temperature=0.6, max_tokens=4096)
