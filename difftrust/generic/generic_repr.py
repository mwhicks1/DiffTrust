import builtins
from collections.abc import Mapping


class ReprHandler:
    """Base class for all repr handlers."""

    def can_handle(self, obj):
        """Determine if this handler can process the object."""
        raise NotImplementedError

    def handle(self, obj, builder, visited):
        """Return the string representation of the object."""
        raise NotImplementedError


class BasicTypeHandler(ReprHandler):
    """Handler for basic built-in types."""

    def can_handle(self, obj):
        return isinstance(obj, (int, float, bool, type(None), str))

    def handle(self, obj, builder, visited):
        return repr(obj)


class SequenceHandler(ReprHandler):
    """Handler for sequence types like list and tuple."""
    brackets = {"list": '[]', 'tuple': '()'}

    def can_handle(self, obj):
        return isinstance(obj, (list, tuple))

    def handle(self, obj, builder, visited):
        if id(obj) in visited:
            return '<...>'
        visited.add(id(obj))
        elements = ', '.join(builder.repr(item, visited) for item in obj)
        visited.remove(id(obj))
        obj_type = type(obj).__name__
        return f"{self.brackets[obj_type][0]}{elements}{self.brackets[obj_type][1]}"


class SetHandler(ReprHandler):
    """Handler for set and frozenset types."""

    def can_handle(self, obj):
        return isinstance(obj, (set, frozenset))

    def handle(self, obj, builder, visited):
        if id(obj) in visited:
            return '<...>'
        visited.add(id(obj))
        elements = ', '.join(builder.repr(item, visited) for item in obj)
        visited.remove(id(obj))
        return f"{{{elements}}}"


class MappingHandler(ReprHandler):
    """Handler for mapping types like dict."""

    def can_handle(self, obj):
        return isinstance(obj, Mapping)

    def handle(self, obj, builder, visited):
        if id(obj) in visited:
            return '<...>'
        visited.add(id(obj))
        items = ', '.join(f"{builder.repr(k, visited)}: {builder.repr(v, visited)}" for k, v in obj.items())
        visited.remove(id(obj))
        return f"{{{items}}}"


class ObjectHandler(ReprHandler):
    """Handler for user-defined objects."""

    def can_handle(self, obj):
        return True  # Fallback handler

    def handle(self, obj, builder, visited):

        if id(obj) in visited:
            return '<...>'
        visited.add(id(obj))
        obj_repr = type(obj).__repr__
        if obj_repr is not builtins.object.__repr__:
            result = repr(obj)
        else:
            try:
                attrs = vars(obj)
                attr_str = ', '.join(f"{key}={builder.repr(value, visited)}" for key, value in attrs.items())
                result = f"{obj.__class__.__name__}({attr_str})"
            except TypeError:
                result = repr(obj)
        visited.remove(id(obj))
        return result


class ReprBuilder:
    """Main class to build string representations of objects."""

    def __init__(self):
        self.handlers = [
            BasicTypeHandler(),
            SequenceHandler(),
            SetHandler(),
            MappingHandler(),
            ObjectHandler()
        ]

    def repr(self, obj, visited=None):
        if visited is None:
            visited = set()
        for handler in self.handlers:
            if handler.can_handle(obj):
                return handler.handle(obj, self, visited)
        return repr(obj)  # Fallback


_builder = ReprBuilder()
generic_repr = _builder.repr
