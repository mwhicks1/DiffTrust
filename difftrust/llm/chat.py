import json
import re
from typing import Callable


def _extract_python_code_blocks(text):
    code_blocks = []
    start_tag = "```python"
    end_tag = "```"
    start = 0

    while True:
        start_idx = text.find(start_tag, start)
        if start_idx == -1:
            break
        start_idx += len(start_tag)
        end_idx = text.find(end_tag, start_idx)
        if end_idx == -1:
            break
        code = text[start_idx:end_idx].strip()
        code_blocks.append(code)
        start = end_idx + len(end_tag)

    return code_blocks


_log_function = lambda x: None


def _set_log_function(log_func: Callable[[str], type(None)]):
    global _log_function
    _log_function = log_func


class Msg:
    def __init__(self, writer: str, content: str):
        self.writer: str = writer
        self.content: str = content

    def __repr__(self):
        return f"{self.writer}: {self.content}"

    def __str__(self):
        return self.__repr__()

    def to_dict(self):
        return {'writer': self.writer, 'content': self.content}

    @staticmethod
    def from_dict(data: dict):
        return Msg(data['writer'], data['content'])

    def extract_code(self):
        codes = []
        for code in _extract_python_code_blocks(self.content):
            codes.append(code)
        assert len(codes) == 1
        return codes[0]

    def extract_codes(self):
        codes = []
        for code in _extract_python_code_blocks(self.content):
            codes.append(code)
        return codes


class Chat:
    def __init__(self, llm, sys_instructions: str):
        self.llm = llm
        self.chatstream: list[Msg] = [Msg("system", sys_instructions)]

    def last(self):
        return self.chatstream[-1].content if self.chatstream else None

    def msg(self, msg: Msg):
        self.chatstream.append(msg)

    def ask(self, content: str):
        msg = Msg('user', content)
        _log_function(msg)
        self.msg(msg)
        return self.run()

    def run(self):
        msg = self.llm.run(self)
        _log_function(msg)
        self.msg(msg)
        return msg

    def save_to_txt(self, filename: str):
        with open(filename, 'w', encoding='utf-8') as f:
            for message in self.chatstream:
                json.dump(message.to_dict(), f)
                f.write('\n')

    def load_from_txt(self, filename: str):
        self.chatstream.clear()
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    self.chatstream.append(Msg.from_dict(data))


def extract_code(text: str):
    codes = []
    for code in _code_extractor.finditer(text):
        codes.append(code.group(1))
    return codes[-1]
