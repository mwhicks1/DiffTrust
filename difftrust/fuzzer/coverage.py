import sys


class LineCoverageTracer:
    def __init__(self):
        self.trace_hash = None
        self.prohibited_files = [
            "pydevd_tracing.py",
            "traceback.py",
            "linecache.py"
        ]

    def _accept_filename(self, filename):
        for file in self.prohibited_files:
            if file in filename:
                return False
        return True

    def tracer(self, frame, event, arg):
        if event == 'line':
            filename = frame.f_code.co_filename
            if self._accept_filename(filename):
                lineno = frame.f_lineno
                self.trace_hash = hash((self.trace_hash, (filename, lineno)))
        return self.tracer

    def run(self, func, *args, **kwargs):
        self.trace_hash = None
        sys.settrace(self.tracer)
        try:
            result = func(*args, **kwargs)
        except Exception as exception:
            result = exception
        finally:
            sys.settrace(None)
        return result, self.trace_hash
