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


# Copyright 2021 The NetKet Authors - All rights reserved.
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


"""
This file contains logic for different decorators responsible for different deprecation warnings

NOTE: part(s) of, or the entire content, of this file is obtained from NetKet's source code
      the original copyright mentioned above applies.
"""

import warnings
import functools
import inspect
from textwrap import dedent


def deprecated(
    reason: str = None,
    func_name: str = None,
):
    """
    Decorator to mark a function or class as deprecated.

    When applied, calling the decorated object will emit a :class:`FutureWarning` indicating that
    the function/class is deprecated and will be removed in a future release. An optional
    ``reason`` can be included to provide migration notes, and an optional ``func_name`` can be
    supplied to override the name shown in the warning (useful for aliases).

    :param reason: Optional additional notes describing why the object is deprecated and/or how to
                   migrate away from it.
    :param func_name: Optional name to display in the warning instead of ``func.__name__``.
    :returns: A decorator that wraps the target callable/class and emits a deprecation warning on
              use.
    """

    def decorator(func):
        object_type = "class" if inspect.isclass(func) else "function"
        function_name = func_name or func.__name__
        message = (
            f"\n\nCall to deprecated {object_type} {function_name!r}\n\n"
            f"{object_type.capitalize()} {function_name!r} is now deprecated and will be "
            f"removed in the next version release.\n\n"
            f"Please update your code to remove usages of "
            f"the {object_type} {function_name!r}. "
            f"If you intend to continue using it, you will have to create your own "
            f"implementation."
        )

        if reason is not None:
            message += f"\n\nNotes:\t{dedent(reason)}"

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(message, category=FutureWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def deprecation_warning(depr_message: str) -> None:
    """
    Emit a :class:`FutureWarning` with a provided deprecation message.

    The message is passed through :func:`textwrap.dedent` and emitted with ``stacklevel=2`` so the
    warning points to the user's call site.

    :param depr_message: Message describing the deprecation and (optionally) the recommended
                         replacement.
    :returns: ``None``.
    """

    warnings.warn(dedent(depr_message), category=FutureWarning, stacklevel=2)


def old_name_deprecation(
    function_name: str,
    reason: str = "",
):
    """
    Decorator factory to deprecate an old function name in favour of a new one.

    When applied, calling the decorated function emits a :class:`FutureWarning` informing the user
    that ``func.__name__`` has been renamed to ``function_name`` and that the old name will be
    removed in a future release. An optional ``reason`` may be appended with additional context.

    :param function_name: The new (preferred) function name to which users should migrate.
    :param reason: Optional additional notes describing the rename and/or migration guidance.
    :returns: A decorator that wraps the function and emits a rename deprecation warning on call.
    """

    def deprecated_decorator(func):
        """
        The actual old name decorator
        """

        @functools.wraps(func)
        def deprecated_function(*args, **kwargs):
            """
            The deprecated function
            """
            warnings.warn(
                dedent(f"""

    {func.__name__} has been renamed to {function_name}. The old name is
    now deprecated and will be removed in the next version release.

    Please update your code by changing occurrences of `{func.__name__}` with
    `{function_name}`.

    {dedent(reason)}

                    """),
                category=FutureWarning,
                stacklevel=2,
            )

            return func(*args, **kwargs)

        return deprecated_function

    return deprecated_decorator


def parameter_name_deprecation(
    old_param: str,
    new_param: str,
    reason: str = "",
):
    """
    Decorator factory to deprecate a keyword parameter name in favour of a new one.

    When applied, calling the decorated function will emit a :class:`FutureWarning` if the user
    supplies ``old_param`` as a keyword argument, indicating that it will be renamed to
    ``new_param`` in a future release. The function call proceeds unchanged (the keyword is not
    automatically renamed).

    :param old_param: Deprecated keyword parameter name.
    :param new_param: Replacement keyword parameter name that users should switch to.
    :param reason: Optional additional notes describing the change and/or migration guidance.
    :returns: A decorator that wraps the function and emits a parameter rename warning when
              ``old_param`` is used.
    """

    def deprecated_decorator(func):
        """
        The actual decorator
        """

        @functools.wraps(func)
        def deprecated_function(*args, **kwargs):
            """
            The function with the deprecated param name
            """
            if old_param in kwargs:
                warnings.warn(
                    dedent(f"""

    The parameter `{old_param}` in function `{func.__name__}` is deprecated and will be renamed to 
    `{new_param}` in a future release.

    Please update your code to use `{new_param}` instead of `{old_param}`.

    {dedent(reason)}

                        """),
                    category=FutureWarning,
                    stacklevel=2,
                )
                # kwargs[new_param] = kwargs.pop(old_param)

            return func(*args, **kwargs)

        return deprecated_function

    return deprecated_decorator
