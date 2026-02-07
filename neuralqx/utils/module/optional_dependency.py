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
This file contains logic for checking for optional dependencies of the package

NOTE: part(s) of, or the entire content, of this file is obtained from NetKet's source code
      the original copyright mentioned above applies.
"""

from types import ModuleType
import importlib


def import_optional_dep(
    name: str,
    minimum_version: str = "",
    reason: str = "",
) -> ModuleType:
    """
    A function that attempts to import a library with the given name. An error is raised if
    unsuccessful

    :param name: the name of the library to be imported
    :param minimum_version: the minimum version required
    :param reason: the reason why this library is required
    """

    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if minimum_version != "":
            minimum_version = f">= {minimum_version}"
        raise ModuleNotFoundError(f"""

            Could not import `{name}`, which is necessary to use
            `{reason}`.

            To install it, run

                pip install {name} {minimum_version}

            """) from exc
