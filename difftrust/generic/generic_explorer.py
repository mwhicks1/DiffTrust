import builtins
from collections.abc import Mapping


class ExplorerHandler:
    """Base class for all repr handlers."""

    def can_handle(self, obj):
        """Determine if this handler can process the object."""
        raise NotImplementedError

    def handle(self, obj, builder):
        """Return the string representation of the object."""
        raise NotImplementedError


class BasicTypeHandler(ExplorerHandler):
    """Handler for basic built-in types."""

    def can_handle(self, obj):
        return isinstance(obj, (int, float, bool, type(None), str))

    def handle(self, obj, builder):
        builder.explored.append(id(obj))


class IterableHandler(ExplorerHandler):
    """Handler for sequence types like list and tuple."""

    def can_handle(self, obj):
        return isinstance(obj, (list, tuple, set, frozenset))

    def handle(self, obj, builder):
        builder.explored.append(id(obj))
        for elt in obj:
            builder.aux_explore(elt)


class MappingHandler(ExplorerHandler):
    """Handler for mapping types like dict."""

    def can_handle(self, obj):
        return isinstance(obj, Mapping)

    def handle(self, obj, builder):
        builder.explored.append(id(obj))
        for key, value in obj.items():
            builder.aux_explore(key)
            builder.aux_explore(value)


class ObjectHandler(ExplorerHandler):
    """Handler for user-defined objects."""

    def can_handle(self, obj):
        return True  # Fallback handler

    def handle(self, obj, builder):
        builder.explored.append(id(obj))
        try:
            attrs = vars(obj)
            builder.aux_explore(attrs)
        except TypeError:
            pass


class ReprBuilder:
    """Main class to build string representations of objects."""

    def __init__(self):
        self.handlers = [
            BasicTypeHandler(),
            IterableHandler(),
            MappingHandler(),
            ObjectHandler()
        ]
        self.explored = set()

    def aux_explore(self, obj):
        for handler in self.handlers:
            if handler.can_handle(obj):
                return handler.handle(obj, self)
        return  # Fallback

    def explore(self, obj):
        self.explored = []
        self.aux_explore(obj)
        return self.explored


_builder = ReprBuilder()
generic_explorer = _builder.explore
