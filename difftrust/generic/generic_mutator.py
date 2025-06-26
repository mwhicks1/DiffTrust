import copy
import random
from difftrust.generic.generic_explorer import generic_explorer


def my_randint(a: int, b: int):
    if a == b:
        return b
    else:
        return random.randint(a, b)


class MutatorHandler:
    """Base class for all repr handlers."""

    def can_handle(self, obj):
        """Determine if this handler can process the object."""
        raise NotImplementedError

    def mutate(self, obj):
        """Mutate the object."""
        raise NotImplementedError

    def handle(self, obj, builder):
        """Handle the object."""
        raise NotImplementedError


class BasicTypeHandler(MutatorHandler):

    def handle(self, obj, builder):

        if builder.current_id > builder.id2mutate:
            return obj

        if builder.current_id == builder.id2mutate:
            builder.current_id += 1
            return self.mutate(obj)

        builder.current_id += 1
        return obj


class IntTypeHandler(BasicTypeHandler):

    def can_handle(self, obj):
        return isinstance(obj, int)

    def mutate(self, obj: int):
        return obj + random.choice([-10, -1, 1, 10])


class FloatTypeHandler(BasicTypeHandler):

    def can_handle(self, obj):
        return isinstance(obj, float)

    def mutate(self, obj: float):
        rnd = random.uniform(-1.0, 1.0)
        rnd2 = random.uniform(-10.0, 10.0)
        return obj + random.choice([rnd, rnd, rnd2])


class BoolTypeHandler(BasicTypeHandler):

    def can_handle(self, obj):
        return isinstance(obj, bool)

    def mutate(self, obj: bool):
        return random.choice([False, True])


class NoneTypeHandler(BasicTypeHandler):

    def can_handle(self, obj):
        return isinstance(obj, type(None))

    def mutate(self, obj: type(None)):
        return None


class StrTypeHandler(BasicTypeHandler):

    def __init__(self):
        self.ranges = [
            (0, 31),  # control chars
            (32, 127),  # printable
            (127, 255)  # extended
        ]

    def rnd_char(self):
        idx = random.choice([0, 0, 1, 1, 1, 1, 2])
        return chr(random.randint(*self.ranges[idx]))

    def can_handle(self, obj):
        return isinstance(obj, str)

    def mutate(self, obj: str):
        lst = list(obj)
        length = len(lst)
        action = random.choice([0, 1, 2, 3, 4, 5, 6])
        if action == 0 and length > 0:
            idx = my_randint(0, length - 1)
            lst[idx] = self.rnd_char()
        elif action == 1 and length > 0:
            idx = my_randint(0, length - 1)
            lst.pop(idx)
        elif action == 2:
            letter = self.rnd_char()
            if length == 0:
                lst.append(letter)
            else:
                idx = my_randint(0, length - 1)
                lst.insert(idx, letter)
        elif action == 3 and length > 0:
            idx = my_randint(0, length - 1)
            idx2 = my_randint(idx, length - 1)
            lst = lst[0:idx] + lst[idx2:length]
        elif action == 4 and length > 0:
            idx = my_randint(0, length - 1)
            idx2 = my_randint(idx, length - 1)
            lst = lst[0:idx2] + lst[idx:idx2] + lst[idx2:length]
        elif action == 5:
            for _ in range(my_randint(0, 10)):
                lst.append(self.rnd_char())
        elif action == 6 and length > 0:
            for _ in range(my_randint(0, length-1)):
                lst.pop()
        return ''.join(lst)


class ListTypeHandler(MutatorHandler):
    """Handler for sequence types like list and tuple."""

    def can_handle(self, obj):
        return isinstance(obj, list)

    def mutate(self, obj):
        length = len(obj)
        obj = [elt for elt in obj]
        action = random.choice([0, 1, 2, 3])
        if action == 0:
            elt = random.choice([None, False, 0, 0.0, "", [], {}, set()])
            if length == 0:
                obj.append(elt)
            else:
                idx = my_randint(0, length - 1)
                obj.insert(idx, elt)
        elif action == 1 and length > 1:
            idx1 = my_randint(0, length - 1)
            idx2 = my_randint(0, length - 1)
            obj[idx1], obj[idx2] = obj[idx2], obj[idx1]
        elif action == 2 and length > 0:
            idx1 = my_randint(0, length - 1)
            idx2 = my_randint(0, length - 1)
            obj.insert(idx2, obj[idx1])
        elif action == 3 and length > 0:
            idx = my_randint(0, length - 1)
            obj.pop(idx)
        return obj

    def handle(self, obj, builder):

        if builder.current_id > builder.id2mutate:
            return obj

        if builder.current_id == builder.id2mutate:
            builder.current_id += 1
            return self.mutate(obj)

        builder.current_id += 1
        return [builder.aux_mutate(elt) for elt in obj]


class TupleTypeHandler(MutatorHandler):
    """Handler for sequence types like list and tuple."""

    def can_handle(self, obj):
        return isinstance(obj, tuple)

    def mutate(self, obj):
        obj = list(obj)
        length = len(obj)
        action = random.choice([0, 1, 2, 3])
        if action == 0:
            elt = random.choice([None, False, 0, 0.0, "", [], {}, set()])
            if length == 0:
                obj.append(elt)
            else:
                idx = my_randint(0, length - 1)
                obj.insert(idx, elt)
        elif action == 1 and length > 1:
            idx1 = my_randint(0, length - 1)
            idx2 = my_randint(0, length - 1)
            obj[idx1], obj[idx2] = obj[idx2], obj[idx1]
        elif action == 2 and length > 0:
            idx1 = my_randint(0, length - 1)
            idx2 = my_randint(0, length - 1)
            obj.insert(idx2, obj[idx1])
        elif action == 3 and length > 0:
            idx = my_randint(0, length - 1)
            obj.pop(idx)
        return tuple(obj)

    def handle(self, obj, builder):

        if builder.current_id > builder.id2mutate:
            return obj

        if builder.current_id == builder.id2mutate:
            builder.current_id += 1
            return self.mutate(obj)

        builder.current_id += 1
        return tuple(builder.aux_mutate(elt) for elt in obj)


class SetTypeHandler(MutatorHandler):
    """Handler for set and frozenset types."""

    def can_handle(self, obj):
        return isinstance(obj, set)

    def mutate(self, obj):
        length = len(obj)
        obj = list(obj)
        action = random.choice([0, 1, 2, 3])
        if action == 0:
            elt = random.choice([None, False, 0, 0.0, ""])
            if length == 0:
                obj.append(elt)
            else:
                idx = my_randint(0, length - 1)
                obj.insert(idx, elt)
        elif action == 1 and length > 0:
            idx = my_randint(0, length - 1)
            obj.pop(idx)
        return set(obj)

    def handle(self, obj, builder):

        if builder.current_id > builder.id2mutate:
            return obj

        if builder.current_id == builder.id2mutate:
            builder.current_id += 1
            return self.mutate(obj)

        builder.current_id += 1
        return set(builder.aux_mutate(elt) for elt in obj)


class DictTypeHandler(MutatorHandler):
    """Handler for mapping types like dict."""

    def __init__(self):
        self.choices = []
        keys = [None, False, 0, 0.0, ""]
        values = [None, False, 0, 0.0, "", [], {}, set()]
        for key in keys:
            for value in values:
                self.choices.append((key, value))

    def can_handle(self, obj):
        return isinstance(obj, dict)

    def mutate(self, obj: dict):
        length = len(obj)
        obj = list(obj.items())
        action = random.choice([0, 1, 2, 3])
        if action == 0:
            elt = random.choice(self.choices)
            if length == 0:
                obj.append(elt)
            else:
                idx = my_randint(0, length - 1)
                obj.insert(idx, elt)
        elif action == 1 and length > 1:
            idx1 = my_randint(0, length - 1)
            idx2 = my_randint(0, length - 1)
            obj[idx1], obj[idx2] = obj[idx2], obj[idx1]
        elif action == 2 and length > 0:
            idx1 = my_randint(0, length - 1)
            idx2 = my_randint(0, length - 1)
            obj.insert(idx2, obj[idx1])
        elif action == 3 and length > 0:
            idx = my_randint(0, length - 1)
            obj.pop(idx)
        return {key: val for key, val in obj}

    def handle(self, obj, builder):

        if builder.current_id > builder.id2mutate:
            return obj

        if builder.current_id == builder.id2mutate:
            builder.current_id += 1
            return self.mutate(obj)

        builder.current_id += 1
        new_obj = {}
        for key, value in obj.items():
            key = builder.aux_mutate(key)
            value = builder.aux_mutate(value)
            new_obj[key] = value
        return new_obj


class DefaultHandler(MutatorHandler):
    """Handler for user-defined objects."""

    def can_handle(self, obj):
        return True  # Fallback handler

    def mutate(self, obj):
        return obj

    def handle(self, obj, builder):

        if builder.current_id > builder.id2mutate:
            return obj

        if builder.current_id == builder.id2mutate:
            builder.current_id += 1
            return self.mutate(obj)

        builder.current_id += 1
        try:
            new_obj = copy.copy(obj)
            attrs = vars(obj)
            for key in attrs.keys():
                setattr(new_obj, key, builder.aux_mutate(attrs[key]))
            return new_obj
        except TypeError:
            return obj


class MutatorBuilder:
    """Main class to build string representations of objects."""

    def __init__(self):
        self.handlers = {
            int: IntTypeHandler(),
            float: FloatTypeHandler(),
            bool: BoolTypeHandler(),
            type(None): NoneTypeHandler(),
            str: StrTypeHandler(),
            list: ListTypeHandler(),
            tuple: TupleTypeHandler(),
            set: SetTypeHandler(),
            dict: DictTypeHandler()
        }
        self.default_handler = DefaultHandler()
        self.current_id = None
        self.id2mutate = None

    def aux_mutate(self, obj):
        try:
            return self.handlers[type(obj)].handle(obj, self)
        except KeyError:
            return self.default_handler.handle(obj, self)

    def mutate(self, obj, iters: int = 1):
        for _ in range(iters):
            length = len(generic_explorer(obj))
            self.current_id = 0
            try:
                self.id2mutate = my_randint(0, length - 1)
            except Exception:
                elt = generic_explorer(obj)
                print(elt)
            obj = self.aux_mutate(obj)
        return obj


_builder = MutatorBuilder()
generic_mutator = _builder.mutate
