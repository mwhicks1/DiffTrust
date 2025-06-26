from typing import Callable

from difftrust.core.specification import Specification
from difftrust.core.checking import timeout_call


class Function:
    def __init__(self, spec: Specification, code: str):
        self.spec = spec
        self.code = code

    def compile(self, check: Callable = None):
        namespace = {}
        try:
            exec(self.code, namespace)
        except Exception as error:
            raise RunningError(self, error)

        try:
            compiled = namespace[self.spec.name]
        except KeyError:
            raise FunctionNameNotFound(self)

        if check is not None:
            if not check(compiled):
                raise CheckFailedError(func=self, check=check)

        return compiled

    def force_compile(self, check: Callable = None, timeout: float = None):
        if timeout is not None:
            try:
                def tmp():
                    exec(self.code, {})

                timeout_call(tmp, (), {}, timeout)
            except TimeoutError:
                return CompilationErrorFunction(self, f"Timeout of {60} s. hit during compilation")

        try:
            return self.compile(check)
        except CompilationError as error:
            return CompilationErrorFunction(self, error.msg)


class CompilationError(Exception):
    def __init__(self, func: Function, msg: str):
        self.func = func
        self.msg = msg
        super().__init__(msg)


class RunningError(CompilationError):
    def __init__(self, func: Function, error: Exception):
        super().__init__(func, f"{type(error).__name__} : {error}")
        self.error = error


class FunctionNameNotFound(CompilationError):
    def __init__(self, func: Function):
        super().__init__(func, f"Function name not found in the code")


class CheckFailedError(CompilationError):
    def __init__(self, func: Function, check: Callable=None):
        self.check = check
        super().__init__(func, f"Function implementation does check property")


class CompilationErrorFunction:
    def __init__(self, func: Function, error: str = None):
        self.func = func
        self.error = error

    def __call__(self, *args, **kwargs):
        raise self.error
