import abc

from difftrust.llm.chat import Chat, Msg


class LLM(abc.ABC):
    def __init__(self, name: str, temperature: float = 0.6):
        self.name = name
        self.temperature = temperature

    @abc.abstractmethod
    def run(self, chat: Chat) -> Msg:
        pass
