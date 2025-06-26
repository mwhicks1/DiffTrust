import copy
import sys
from typing import Any, Dict, List, Set
import pathlib
from deepdiff import DeepDiff
from difftrust.tracing.events import ExecEvent, FunctionCall, FunctionReturn, ValueChanged, LineExec, ExceptionRaised


def safe_diff(prev_locals, curr_locals):
    return DeepDiff(prev_locals, curr_locals, ignore_order=True, verbose_level=2)


def safe_copy(obj: Any) -> Any:
    try:
        return copy.deepcopy(obj)
    except Exception:
        return obj


class Trace:
    def __init__(self):
        self.exec_events: List[ExecEvent] = []

    def append(self, exec_event: ExecEvent):
        self.exec_events.append(exec_event)

    def filter_by_file(self, selected_files: Set[str], keep_exception: bool = True) -> 'Trace':
        selected_files = {pathlib.Path(file).resolve().as_posix() for file in selected_files}
        file_stack = []
        filtered_trace = Trace()

        for event in self.exec_events:
            if isinstance(event, ExceptionRaised) and keep_exception:
                filtered_trace.append(event)
            elif isinstance(event, FunctionCall):
                file_stack.append(event.filename)
                if event.filename in selected_files:
                    filtered_trace.append(event)
            elif isinstance(event, FunctionReturn):
                if file_stack:
                    file_stack.pop()
                if not file_stack or file_stack[-1] in selected_files:
                    filtered_trace.append(event)
            else:
                if not file_stack or file_stack[-1] in selected_files:
                    filtered_trace.append(event)

        return filtered_trace

    def __repr__(self):
        return '\n'.join(str(event) for event in self.exec_events)


class Tracer:
    def __init__(self, selected_files: set[str] = None, selected_functions: set[str] = None):
        self.locals: Dict[int, Dict[str, Any]] = {}
        self.call_stack: List[str] = []
        self.file_stack: List[str] = []
        self.trace = Trace()
        self.file_cache: Dict[str, List[str]] = {}
        self.selected_files = selected_files
        if self.selected_files is not None:
            self.selected_files = {pathlib.Path(file).resolve().as_posix() for file in self.selected_files}
        self.selected_functions = selected_functions

    def _get_line(self, filename: str, lineno: int) -> str:
        if filename not in self.file_cache:
            with open(filename, 'r') as file:
                self.file_cache[filename] = file.readlines()
        return self.file_cache[filename][lineno - 1].strip()

    def _frame_info(self, frame):
        code = frame.f_code
        func_name = code.co_name
        lineno = frame.f_lineno
        filename = pathlib.Path(code.co_filename).resolve().as_posix()
        line = self._get_line(filename, lineno)
        return code, func_name, lineno, filename, line

    def _allowed_file(self, frame, event, arg, frame_info):

        if self.selected_files is not None:

            # We are currently in an allowed file
            allowed_case_0 = len(self.file_stack) == 0 or self.file_stack[-1] in self.selected_files
            # If it is a return that will return into an allowed file we accept
            allowed_case_1 = event == 'return' and (
                    len(self.file_stack) < 2 or self.file_stack[-2] in self.selected_files)

            return allowed_case_0 or allowed_case_1

        else:

            return True

    def _allowed_function(self, frame, event, arg, frame_info):

        # Filtering with respect to selected_functions
        if self.selected_functions is not None:

            # We are currently in an allowed function
            allowed_case_0 = len(self.call_stack) == 0 or self.call_stack[-1] in self.selected_functions
            # If it is a return that will return into an allowed file we accept
            allowed_case_1 = event == 'return' and (
                    len(self.call_stack) < 2 or self.call_stack[-2] in self.selected_functions)

            return allowed_case_0 or allowed_case_1

        else:

            return True

    def _trace_diff(self, frame, event, arg, frame_info):

        # Comparing the previous locals states with the new one
        curr_locals = frame.f_locals
        prev_locals = self.locals.get(id(frame), {})
        diff = safe_diff(prev_locals, curr_locals)
        if diff:
            self.trace.append(ValueChanged(diff.to_dict()))

        # Saving the new locals state
        self.locals[id(frame)] = safe_copy(curr_locals)

    def _trace_call(self, frame, event, arg, frame_info):

        code, func_name, lineno, filename, line = frame_info  # Extracting frame information
        args = {var: frame.f_locals.get(var) for var in code.co_varnames[:code.co_argcount]}  # Getting arguments

        self.locals[id(frame)] = {var: safe_copy(value) for var, value in args.items()}  # Updating the locals
        self.trace.append(FunctionCall(func_name, filename, lineno, args))  # Saving the call in the trace

        self.call_stack.append(func_name)  # Saving information in the stack
        self.file_stack.append(filename)  # Saving information in the stack

    def _trace_line(self, frame, event, arg, frame_info):

        code, func_name, lineno, filename, line = frame_info  # Extracting frame information

        self.trace.append(LineExec(func_name, filename, lineno, line))  # Saving the line exec in the trace

    def _trace_return(self, frame, event, arg, frame_info):

        code, func_name, lineno, filename, line = frame_info  # Extracting frame information

        self.trace.append(FunctionReturn(func_name, arg))  # Saving the return in the trace

        if self.call_stack:
            self.call_stack.pop()  # Pop the call stack
            self.file_stack.pop()  # Pop the file call stack

    def _trace_exception(self, frame, event, arg, frame_info):
        code, func_name, lineno, filename, line = frame_info
        exc_type, exc_value, exc_tb = arg
        self.trace.append(ExceptionRaised(func_name, filename, lineno, exc_type, exc_value, exc_tb))

    def _trace_all(self, frame, event, arg, frame_info):

        if event == 'call':
            self._trace_call(frame, event, arg, frame_info)
        elif event == 'line':
            self._trace_diff(frame, event, arg, frame_info)
            self._trace_line(frame, event, arg, frame_info)
        elif event == 'return':
            self._trace_return(frame, event, arg, frame_info)
        elif event == 'exception':
            self._trace_exception(frame, event, arg, frame_info)

    def _trace(self, frame, event: str, arg):

        frame_info = self._frame_info(frame)
        code, func_name, lineno, filename, line = frame_info

        # Avoiding tracing during the stop function
        if func_name == 'stop' and filename == pathlib.Path(__file__).resolve().as_posix():
            return None

        # Filtering with respect to selected_files and selected_functions
        allowed_file = self._allowed_file(frame, event, arg, frame_info)
        allowed_function = self._allowed_function(frame, event, arg, frame_info)
        if not (allowed_file and allowed_function):
            return None

        # Tracing
        self._trace_all(frame, event, arg, frame_info)

        return self._trace

    def start(self):
        sys.settrace(self._trace)

    def stop(self):
        sys.settrace(None)

    def reset(self):
        self.locals: Dict[int, Dict[str, Any]] = {}
        self.call_stack: List[str] = []
        self.file_stack: List[str] = []
        self.trace = Trace()
        self.file_cache: Dict[str, List[str]] = {}


if __name__ == '__main__':
    import random


    def foo():
        tab = []
        for _ in range(4):
            tab.append({random.randint(0, 10): random.randint(0, 10)})
        tab[0]['toto'] = random.randint(0, -1)
        tab[0] = tab[6]  # Will raise IndexError
        return False


    tracer = Tracer()
    tracer.start()
    try:
        foo()
    except Exception as e:
        print(type(e).__name__, ':', e)
    tracer.stop()
    print(tracer.trace)
    print("-"*100)
    tracer = Tracer(selected_files={__file__})
    tracer.start()
    try:
        foo()
    except Exception as e:
        print(type(e).__name__, ':', e)
    tracer.stop()
    print(tracer.trace)
    print("-" * 100)
    tracer = Tracer(selected_functions={'foo'})
    tracer.start()
    try:
        foo()
    except Exception as e:
        print(type(e).__name__, ':', e)
    tracer.stop()
    print(tracer.trace)
