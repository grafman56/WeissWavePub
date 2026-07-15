"""Content signature of the code that computes cached values.

Both caches were keyed on a hand-maintained SCHEMA version whose stated
contract is "bump when a column is added or renamed". That contract does not
cover the failure that actually happened. Fixing rising() in 655cd25 changed
the VALUE of every volume column and the NAME of none of them, so
SIGNALS_SCHEMA_VERSION was still honestly 3 and a pre-fix cache was still a
legitimate hit. The rebuild only happened because that commit incidentally
changed the key's filename shape. The next value-only correction would not be
so lucky.

Hashing the source that produces the values closes it: any edit to the
computation invalidates every cache derived from it, with nobody needing to
remember. The asymmetry is the whole argument -- over-invalidating costs one
rebuild, under-invalidating costs however long it takes to notice that plausible
numbers are wrong.

Reachability is resolved from the source rather than from a list of modules,
because a list of modules is the same kind of thing as a version constant
someone has to remember to bump. Callers name only ROOTS (the entry points they
call into); everything those roots import is picked up automatically, so a new
indicator module is covered the moment it is imported.
"""

import ast
import hashlib
import os
from functools import lru_cache

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))


def _module_src(name):
    """Source bytes of `name` inside this package, or None if it isn't here.

    Absent is a legitimate answer, not an error: combined.py is proprietary and
    excluded from public distributions, and a build without it computes
    genuinely different signals -- which is exactly what a differing hash should
    say.
    """
    path = os.path.join(_PKG_DIR, name + ".py")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return f.read()


def _relative_imports(tree):
    """Module names pulled in by a relative import, at any nesting depth.

    Walks the whole tree rather than just module level: signals.py imports
    .combined inside a try/except, and a top-level-only scan would silently drop
    it from the graph -- the same class of miss this module exists to prevent.
    """
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level:
            if node.module:                    # from .core import ema
                found.add(node.module.split(".")[0])
            else:                              # from . import core
                found.update(alias.name for alias in node.names)
    return found


@lru_cache(maxsize=None)
def package_sig(*roots):
    """Short hash of `roots` and every module in this package they reach.

    Only this package is walked. numpy/pandas versions are a real input to the
    numbers too, but hashing them means hashing site-packages; pin them if that
    matters.

    Memoized per process: a run uses the code it started with, and re-reading
    the graph on every cache-key construction would show up in sweeps.
    """
    seen = {}
    stack = list(roots)
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        src = _module_src(name)
        if src is None:
            continue
        seen[name] = src
        stack.extend(_relative_imports(ast.parse(src, filename=name)))

    h = hashlib.sha1()
    for name in sorted(seen):
        h.update(name.encode())
        h.update(seen[name])
    return h.hexdigest()[:8]
