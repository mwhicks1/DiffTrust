import random


def _lst_str_additional_information(additional_information: dict[str, str]):
    if additional_information is None:
        return None
    lst = []
    for key, content in additional_information.items():
        content = content.replace('\n', '\n\t\t')
        lst.append(f"{key} : \n\n\t\t{content}")
    return lst


class Specification:
    """
    Represents a detailed specification for a function, including its name, signature,
    description, required imports, constraints, and any additional information.

    This class is useful for generating structured, readable documentation or prompts
    based on the function's metadata.

    Attributes:
        name (str): The name of the function.
        signature (str): The function's signature excluding the name (e.g., '(x: int) -> int').
        description (str): A multiline description of the function's behavior and purpose.
        imports (list[str], optional): A list of required import statements.
        constraints (list[str], optional): A list of implementation constraints or rules.
        additional_information (dict[str, str], optional): Additional labeled information
            relevant to the function, such as usage notes or implementation tips.

    Methods:
        __repr__(): Returns a formatted string representing the full function specification.
    """

    def __init__(
            self,
            name: str,
            signature: str,
            description: str,
            imports: list[str] = None,
            constraints: list[str] = None,
            additional_information: dict[str, str] = None
    ):
        self.name: str = name
        self.signature: str = signature
        self.description: str = description
        self.imports: list[str] = imports
        self.constraints: list[str] = constraints
        self.additional_information: list[str] = _lst_str_additional_information(additional_information)

    def __repr__(self):
        description = self.description.replace('\n', '\n\t')
        prompt_parts = [
            "** Function specification **\n",
            f"Name:\n\t{self.name}\n",
            f"Signature:\n\t{self.name}{self.signature}\n",
            f"Description:\n\t{description}\n"
        ]

        if self.imports:
            imports_str = '\n\t'.join(self.imports)
            prompt_parts.append(f"Required Imports:\n\t{imports_str}\n")

        if self.constraints:
            constraints_str = '\n\t'.join(f"- {c}" for c in self.constraints)
            prompt_parts.append(f"Constraints:\n\t{constraints_str}\n")

        if self.additional_information:
            additional_information_str = '\n\t'.join(f"- {info}" for info in self.additional_information)
            prompt_parts.append(f"Additional information:\n\t{additional_information_str}\n")

        return '\n'.join(prompt_parts)

    def gen_spec(self):
        """Return a specification for a generator of input for the function in this specification"""

        return Specification(
            name=f"gen_{self.name}",
            signature=f"() -> tuple[\"\"\" place here the inputs types of {self.name} \"\"\"]",
            description=f""
                        f"This function generates randomly a correct input tuple for {self.name} in the order.\n"
                        f"The specification of {self.name} is : \n"
                        f"{self}",
            constraints=[
                "You can use the module random"
            ]
        )


class ProbabilisticSpecification:
    def __init__(self, spec_list: list[tuple[Specification, int]]):
        self.spec_list = spec_list

    def sample(self) -> Specification:
        total = sum(score for _, score in self.spec_list)
        rnd = random.randrange(0, total)
        count = 0
        for spec, score in self.spec_list:
            count += score
            if count >= rnd:
                return spec
        assert 0