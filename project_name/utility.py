import inspect
import json
from pathlib import Path
from typing import Any, Callable, Dict, NamedTuple


def namedtuple_to_dict(o: NamedTuple):
    return {
        k: v if not hasattr(v, '_asdict') else namedtuple_to_dict(v)
        for k, v in o._asdict().items()
    }


class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Path):
            return str(o)
        if hasattr(o, '_asdict'):
            return o._asdict()
        return json.JSONEncoder.default(self, o)


def save_arguments(func: Callable, arguments: Dict[str, Any], path: Path):
    args = inspect.getfullargspec(func).args
    obj = {k: v for k, v in arguments.items() if k in args}
    json.dump(obj, path.open('w'), indent=2, sort_keys=True, cls=JSONEncoder)
