# SPDX-License-Identifier: Apache-2.0
"""Compatibility helpers for renamed dataclass fields."""

import functools


def deprecated_init_kwargs(**aliases):
    """Let a dataclass constructor accept renamed keyword arguments.

    A read-write ``property`` alias covers attribute access
    (``cfg.legacy = v`` / ``cfg.legacy``) but not the constructor: the
    generated ``__init__`` only knows the canonical field names, so
    ``Cls(legacy=...)`` and ``dataclasses.replace(cfg, legacy=...)``
    raise ``TypeError``. This decorator wraps the generated ``__init__``
    and maps each deprecated keyword onto its canonical one.

    Apply it *above* ``@dataclass``::

        @deprecated_init_kwargs(old_name="new_name")
        @dataclass
        class Cfg:
            new_name: str = "fp32"

    An explicitly passed deprecated keyword (not ``None``) overrides the
    canonical keyword. That precedence is deliberate: ``dataclasses.replace``
    re-materializes every init field from the current instance, so a
    canonical-first ``setdefault`` would silently swallow
    ``replace(cfg, legacy=...)``. Dict-shaped surfaces (settings files,
    env, API requests) keep their canonical-first precedence in their own
    parsing code; a constructor keyword is always a deliberate selection.
    """

    def decorator(cls):
        orig_init = cls.__init__

        @functools.wraps(orig_init)
        def __init__(self, *args, **kwargs):  # noqa: N807
            for legacy, canonical in aliases.items():
                if legacy in kwargs:
                    value = kwargs.pop(legacy)
                    if value is not None:
                        kwargs[canonical] = value
            orig_init(self, *args, **kwargs)

        cls.__init__ = __init__
        return cls

    return decorator
