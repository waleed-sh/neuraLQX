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

"""
Experimental decorator implementation
"""

import functools

from typing import Callable
from typing import Type
from typing import Any

from neuralqx import cfg
from neuralqx.utils.errors import DeniedExperimentalFeatureError


def experimental(obj: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator to mark functions or classes as experimental.

    When an experimental function or class is used, it raises a RuntimeError
    unless the 'EXPERIMENTAL' configuration variable is set to True.

    :param obj: the function or class to be decorated.
    :type obj: Callable or Type
    :return: the decorated function or class
    :rtype: Callable or Type
    :raises RuntimeError: if experimental features are not enabled
    """

    if isinstance(obj, type):
        # if the object is a class, wrap its __init__ method
        original_init = obj.__init__

        @functools.wraps(original_init)
        def wrapped_init(self, *args: Any, **kwargs: Any) -> None:
            if not cfg.get("EXPERIMENTAL"):
                raise DeniedExperimentalFeatureError(obj.__name__)
            original_init(self, *args, **kwargs)

        obj.__init__ = wrapped_init
        return obj

    elif callable(obj):
        # If the object is a function, wrap it with a new function
        @functools.wraps(obj)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not cfg.get("EXPERIMENTAL"):
                raise DeniedExperimentalFeatureError(obj.__name__ + "()")
            return obj(*args, **kwargs)

        return wrapper

    else:
        raise TypeError(
            "The @experimental decorator can only be applied to functions and classes."
        )
