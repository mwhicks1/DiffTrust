import traceback
import pathlib
from typing import Any, Dict
from difftrust.generic import generic_repr


class ExecEvent:
    """Base class for execution log entries."""
    pass


class FunctionCall(ExecEvent):
    def __init__(self, func_name: str, filename: str, lineno: int, arguments: Dict[str, Any]):
        self.func_name = func_name
        self.filename = filename
        self.lineno = lineno
        self.arguments = generic_repr(arguments)

    def __repr__(self):
        return (f"--> Call to function '{self.func_name}' in {self.filename}:{self.lineno} with arguments:\n"
                f"        {self.arguments}")


class LineExec(ExecEvent):
    def __init__(self, func_name: str, filename: str, lineno: int, line: str):
        self.func_name = func_name
        self.filename = filename
        self.lineno = lineno
        self.line = line

    def __repr__(self):
        file_name = pathlib.Path(self.filename).name
        return f"    Executing line {self.lineno} in function '{self.func_name}' ({file_name}): {self.line}"


class ValueChanged(ExecEvent):
    def __init__(self, changes: Dict[str, Any]):
        self.changes = changes

    def __repr__(self):
        if not self.changes:
            return "        No changes in local variables."
        changes_str = '\n'.join(f"            {key}: {value}" for key, value in self.changes.items())
        return f"        Changes in local variables:\n{changes_str}"


class FunctionReturn(ExecEvent):
    def __init__(self, func_name: str, return_value: Any):
        self.func_name = func_name
        self.return_value = return_value

    def __repr__(self):
        return f"<-- Return from function '{self.func_name}' with value: {repr(self.return_value)}"


class ExceptionRaised(ExecEvent):
    def __init__(self, func_name: str, filename: str, lineno: int, exc_type, exc_value, exc_tb):
        self.func_name = func_name
        self.filename = filename
        self.lineno = lineno
        self.exc_type = exc_type
        self.exc_value = exc_value
        self.exc_tb = exc_tb

    def __repr__(self):
        tb_str = ''.join(traceback.format_exception(self.exc_type, self.exc_value, self.exc_tb)).strip()
        return f"!! Exception in function '{self.func_name}' at {self.filename}:{self.lineno}:\n{tb_str}"
