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


def get_modifier_value(mod: int, max_terms: int) -> str:
    """
    Format an integer modifier as a zero-padded string with a consistent width.

    The returned width is determined by the number of digits of ``max_terms``. This ensures that
    all modifier strings produced with the same ``max_terms`` have the same length and can be
    concatenated safely into a single identifier.

    Example:
        If ``max_terms == 120`` (width 3), then ``mod == 7`` becomes ``"007"`` and ``mod == 42``
        becomes ``"042"``.

    :param mod: Integer modifier to format.
    :param max_terms: Reference value that determines the output width via ``len(str(max_terms))``.
    :returns: Zero-padded string representation of ``mod``.
    """

    return str(mod).zfill(len(str(max_terms)))


def get_modifier_string(*args, max_terms: int) -> str:
    """
    Build a concatenated modifier string from multiple integer modifiers.

    Each modifier in ``args`` is formatted using :func:`get_modifier_value` so that all components
    have the same fixed width determined by ``max_terms`` (i.e. ``len(str(max_terms))``). The final
    string is the concatenation of these fixed-width components.

    :param args: One or more integer modifiers to include in the final string.
    :param max_terms: Reference value that determines the width of each modifier component.
    :returns: Concatenated fixed-width modifier string.
    """

    mod_list = []
    for a in args:
        mod_list.append(get_modifier_value(a, max_terms))
    return "".join(mod_list)
