import inspect

from difftrust.core.specification import Specification, ProbabilisticSpecification
from difftrust.llm.abstract import LLM
from difftrust.llm.chat import Chat, Msg


class SpecificationRefiner:
    def __init__(self, llm: LLM):
        self.llm = llm

    def refine(self, spec: Specification):
        # Starting the chat
        chat = Chat(self.llm, "You are world class python programmer")

        # Identifying the ambiguous elements of the specifications
        question = (
            f"What are the ambiguous elements in this function specification ? \n\n"
            f"{spec}\n\n"
            f"Please be exhaustive..."
        )
        chat.ask(question)

        question = (
            f"What are all the different possible variations of this specification ? "
            f"Please argument about the likelihood of each variation. "
            f"You can annotate each variation with a percentage of likelihood, the greater the percentage the most "
            f"likely is the variation."
        )
        chat.ask(question)

        question = (
            f"Here is the source code of the Specification class : \n\n"
            f"{inspect.getsource(Specification)}\n\n"
            f"Can you write a python code that create a list of tuple[FunctionSpecification, int] representing "
            f"possibles specifications and a score between 0 and 100 representing the likelihood"
            f" (0: very unlikely, 100: very likely) ?\n"
            f"Please write the full code in one go and strictly follow this format :\n"
            f"```python\n"
            f"from difftrust.core.specification import Specification\n"
            f"\n"
            f"spec_list = []\n"
            f"\n"
            f"spec = Specification(\"\"\"Fill here by the specification\"\"\")\n"
            f"score = \"\"\"Fill here by a score from 0 (if the spec is very unlikely) to 100 (if the spec is very "
            f"likely)\"\"\"\n"
            f"spec_list.append((spec, score))\n"
            f"\"\"\"Repeat for each specification ... \"\"\"\n"
            f"\n"
            f"# End code\n"
            f"```\n"
        )
        code = chat.ask(question).extract_code()
        namespace = {}
        exec(code, namespace)
        return ProbabilisticSpecification(namespace["spec_list"])
