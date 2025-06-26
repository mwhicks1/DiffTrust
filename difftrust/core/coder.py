from concurrent.futures import ThreadPoolExecutor

from difftrust.llm.abstract import LLM
from difftrust.llm.chat import Chat
from difftrust.core.function import Function
from difftrust.core.specification import Specification, ProbabilisticSpecification


class Coder:
    def __init__(self, llm: LLM):
        self.llm = llm

    def code(self, spec: Specification):
        chat = Chat(self.llm, "You are world class python programmer")

        question = (
            f"Can you write a function with the following specification:\n\n"
            f"{spec}\n\n"
            f"Please write the full code in one go like this :\n\n"
            f"```python\n"
            f"# imports ...\n"
            f"# Auxiliary functions and classes ...\n"
            f"def {spec.name}{spec.signature}:\n"
            f"    # {spec.name} code ...\n"
            f"# End of python code\n"
            f"```\n\n"
            f"Please ensure the function adheres strictly to the provided specifications.\n"
        )
        codes = chat.ask(question).extract_codes()
        for code in codes:
            if f"def {spec.name}" in code:
                return Function(spec, code)
        return Function(spec, codes[0])

    def sample(self, spec: Specification | ProbabilisticSpecification, samples: int):
        if isinstance(spec, Specification):
            specs = [spec for _ in range(samples)]
        elif isinstance(spec, ProbabilisticSpecification):
            specs = [spec.sample() for _ in range(samples)]
        else:
            assert 0

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.code, specification) for specification in specs]
            return [future.result() for future in futures]

