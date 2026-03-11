# Copyright (c) 2026 The neuraLQX Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations

from contextlib import contextmanager
from functools import wraps

from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Tuple

from ._cfg import get_bool
from ._cfg import get_int
from ._cfg import get_str
from ._jax import block_until_ready
from ._pytrace import python_call_trace
from .profiler import get_profiler


def section(
    name: str,
    cat: str = "",
    *,
    args: Optional[Dict[str, Any]] = None,
    flops: float = 0.0,
    bytes: float = 0.0,
):
    """
    Context manager to profile a section of code.

    :param name: Section name
    :param cat: Category (e.g. "vmc", "sampling", "jax", ...)
    :param args: Additional metadata to attach to trace events
    :param flops: Optional counters for roofline-style derived metrics
    :param bytes: Optional counters for roofline-style derived metrics
    """
    return get_profiler().section(name, cat=cat, args=args, flops=flops, bytes=bytes)


def step(
    step_num: int,
    name: str = "step",
    cat: str = "step",
    *,
    args: Optional[Dict[str, Any]] = None,
):
    """
    Context manager to mark a repeated step (e.g. VMC iteration number).
    """
    return get_profiler().step(step_num=step_num, name=name, cat=cat, args=args)


def profile(
    name: Optional[str] = None,
    cat: str = "",
    *,
    sync: Optional[bool] = None,
    args: Optional[Dict[str, Any]] = None,
    flops: float = 0.0,
    bytes: float = 0.0,
):
    """
    Decorator to profile a function.

    sync:
      If True, block_until_ready() the returned value inside the timed region.
      This provides device-time-ish measurements for JAX, but can introduce delays.
      If None, uses global config (default False).
    """

    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        disp = name or getattr(fn, "__qualname__", getattr(fn, "__name__", "fn"))

        @wraps(fn)
        def wrapped(*a, **k):
            return _run_profiled(
                fn,
                a,
                k,
                name=disp,
                cat=cat,
                sync=sync,
                args=args,
                flops=flops,
                bytes=bytes,
                deep=False,
            )

        return wrapped

    return deco


def _split_prefixes(raw: str) -> Optional[Tuple[str, ...]]:
    items = []
    for item in raw.split(","):
        p = item.strip()
        if p:
            items.append(p)
    return tuple(items) if items else None


def _resolve_deep_opts(
    *,
    deep: Optional[bool],
    deep_include: Optional[Sequence[str]],
    deep_exclude: Optional[Sequence[str]],
    deep_max_depth: Optional[int],
) -> tuple[bool, Optional[Tuple[str, ...]], Optional[Tuple[str, ...]], Optional[int]]:
    if deep is None:
        deep = get_bool(
            "PROFILE_PY_CALLS",
            default=False,
            env_key="NQX_PROFILE_PY_CALLS",
        )

    include: Optional[Tuple[str, ...]]
    exclude: Optional[Tuple[str, ...]]

    if deep_include is not None:
        include = tuple(str(x).strip() for x in deep_include if str(x).strip()) or None
    else:
        include = _split_prefixes(
            get_str(
                "PROFILE_PY_CALLS_INCLUDE",
                default="netket,neuralqx",
                env_key="NQX_PROFILE_PY_CALLS_INCLUDE",
            )
        )

    if deep_exclude is not None:
        exclude = tuple(str(x).strip() for x in deep_exclude if str(x).strip()) or None
    else:
        exclude = _split_prefixes(
            get_str(
                "PROFILE_PY_CALLS_EXCLUDE",
                default="neuralqx.profile",
                env_key="NQX_PROFILE_PY_CALLS_EXCLUDE",
            )
        )

    if deep_max_depth is None:
        deep_max_depth = get_int(
            "PROFILE_PY_CALLS_MAX_DEPTH",
            default=6,
            env_key="NQX_PROFILE_PY_CALLS_MAX_DEPTH",
        )
    if deep_max_depth is not None and int(deep_max_depth) <= 0:
        deep_max_depth = None

    return bool(deep), include, exclude, deep_max_depth


def _run_profiled(
    fn: Callable[..., Any],
    a: tuple[Any, ...],
    k: dict[str, Any],
    *,
    name: str,
    cat: str,
    sync: Optional[bool],
    args: Optional[Dict[str, Any]],
    flops: float,
    bytes: float,
    deep: Optional[bool],
    deep_cat: str = "python",
    deep_include: Optional[Sequence[str]] = None,
    deep_exclude: Optional[Sequence[str]] = None,
    deep_max_depth: Optional[int] = None,
) -> Any:
    prof = get_profiler()
    if not prof.enabled():
        return fn(*a, **k)

    do_sync = bool(sync) if sync is not None else bool(prof.sync_enabled())
    if deep is False:
        do_deep = False
        include = None
        exclude = None
        max_depth = None
    else:
        do_deep, include, exclude, max_depth = _resolve_deep_opts(
            deep=deep,
            deep_include=deep_include,
            deep_exclude=deep_exclude,
            deep_max_depth=deep_max_depth,
        )

    cm = prof.section(name, cat=cat, args=args, flops=flops, bytes=bytes)
    cm.__enter__()
    try:
        if do_deep:
            with python_call_trace(
                cat=deep_cat,
                include_prefixes=include,
                exclude_prefixes=exclude,
                max_depth=max_depth,
            ):
                out = fn(*a, **k)
        else:
            out = fn(*a, **k)

        if do_sync:
            out = block_until_ready(out)
        cm.__exit__(None, None, None)
        return out
    except BaseException as e:
        cm.__exit__(type(e), e, e.__traceback__)
        raise


def wrap_callable(
    fn: Callable[..., Any],
    *,
    name: Optional[str] = None,
    cat: str = "",
    sync: Optional[bool] = None,
    args: Optional[Dict[str, Any]] = None,
    flops: float = 0.0,
    bytes: float = 0.0,
    deep: Optional[bool] = None,
    deep_cat: str = "python",
    deep_include: Optional[Sequence[str]] = None,
    deep_exclude: Optional[Sequence[str]] = None,
    deep_max_depth: Optional[int] = None,
) -> Callable[..., Any]:
    """
    Wrap an arbitrary callable with profiling sections.

    Useful to instrument external library functions without editing their source.
    """
    disp = name or getattr(fn, "__qualname__", getattr(fn, "__name__", "fn"))

    @wraps(fn)
    def wrapped(*a, **k):
        return _run_profiled(
            fn,
            a,
            k,
            name=disp,
            cat=cat,
            sync=sync,
            args=args,
            flops=flops,
            bytes=bytes,
            deep=deep,
            deep_cat=deep_cat,
            deep_include=deep_include,
            deep_exclude=deep_exclude,
            deep_max_depth=deep_max_depth,
        )

    return wrapped


def profile_call(
    fn: Callable[..., Any],
    *a,
    name: Optional[str] = None,
    cat: str = "",
    sync: Optional[bool] = None,
    args: Optional[Dict[str, Any]] = None,
    flops: float = 0.0,
    bytes: float = 0.0,
    deep: Optional[bool] = None,
    deep_cat: str = "python",
    deep_include: Optional[Sequence[str]] = None,
    deep_exclude: Optional[Sequence[str]] = None,
    deep_max_depth: Optional[int] = None,
    **k,
) -> Any:
    """
    Execute one profiled callable invocation.
    """
    disp = name or getattr(fn, "__qualname__", getattr(fn, "__name__", "fn"))
    return _run_profiled(
        fn,
        a,
        k,
        name=disp,
        cat=cat,
        sync=sync,
        args=args,
        flops=flops,
        bytes=bytes,
        deep=deep,
        deep_cat=deep_cat,
        deep_include=deep_include,
        deep_exclude=deep_exclude,
        deep_max_depth=deep_max_depth,
    )


@contextmanager
def patch_method(
    obj: Any,
    method_name: str,
    *,
    name: Optional[str] = None,
    cat: str = "",
    sync: Optional[bool] = None,
    args: Optional[Dict[str, Any]] = None,
    flops: float = 0.0,
    bytes: float = 0.0,
    deep: Optional[bool] = None,
    deep_cat: str = "python",
    deep_include: Optional[Sequence[str]] = None,
    deep_exclude: Optional[Sequence[str]] = None,
    deep_max_depth: Optional[int] = None,
):
    """
    Temporarily patch ``obj.method_name`` with a profiled wrapper.
    """
    with patch_attr(
        obj,
        method_name,
        name=name or f"{type(obj).__name__}.{method_name}",
        cat=cat,
        sync=sync,
        args=args,
        flops=flops,
        bytes=bytes,
        deep=deep,
        deep_cat=deep_cat,
        deep_include=deep_include,
        deep_exclude=deep_exclude,
        deep_max_depth=deep_max_depth,
    ) as wrapped:
        yield wrapped


@contextmanager
def patch_attr(
    target: Any,
    attr_path: str,
    *,
    name: Optional[str] = None,
    cat: str = "",
    sync: Optional[bool] = None,
    args: Optional[Dict[str, Any]] = None,
    flops: float = 0.0,
    bytes: float = 0.0,
    deep: Optional[bool] = None,
    deep_cat: str = "python",
    deep_include: Optional[Sequence[str]] = None,
    deep_exclude: Optional[Sequence[str]] = None,
    deep_max_depth: Optional[int] = None,
):
    """
    Temporarily patch a callable attribute path on an object or module.

    ``attr_path`` can be dotted, for example ``"sampler.sample"``.
    """
    if target is None:
        raise ValueError("target cannot be None")
    if not attr_path:
        raise ValueError("attr_path cannot be empty")

    parts = [p for p in str(attr_path).split(".") if p]
    if not parts:
        raise ValueError("attr_path cannot be empty")

    owner = target
    for part in parts[:-1]:
        owner = getattr(owner, part)
    leaf = parts[-1]

    owner_dict = getattr(owner, "__dict__", None)
    had_local_attr = False
    try:
        if owner_dict is not None:
            had_local_attr = leaf in owner_dict
    except Exception:
        had_local_attr = False

    original = getattr(owner, leaf)
    wrapped = wrap_callable(
        original,
        name=name or attr_path,
        cat=cat,
        sync=sync,
        args=args,
        flops=flops,
        bytes=bytes,
        deep=deep,
        deep_cat=deep_cat,
        deep_include=deep_include,
        deep_exclude=deep_exclude,
        deep_max_depth=deep_max_depth,
    )

    setattr(owner, leaf, wrapped)
    try:
        yield wrapped
    finally:
        if had_local_attr:
            setattr(owner, leaf, original)
        else:
            try:
                delattr(owner, leaf)
            except Exception:
                setattr(owner, leaf, original)


__all__ = [
    "section",
    "step",
    "profile",
    "wrap_callable",
    "profile_call",
    "patch_method",
    "patch_attr",
    "python_call_trace",
]
