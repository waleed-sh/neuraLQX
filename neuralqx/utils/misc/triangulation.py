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
This file includes logic for a class which holds the triangulation terms of the 4D TRC
"""


class Triangulation:  # pylint: disable=R0903
    """
    A primitive class which holds the triangulation data in the 4D Euclidean WCL model for which the
    TRC would act on
    """

    def __init__(
        self,
        **kwargs,
    ) -> None:
        """
        Initializer
        """
        for key, val in kwargs.items():
            setattr(self, key, val)

    def _create_getter(self, attr_name):
        """
        A method which creates a getter for a given attribute
        """

        def getter(self):  # pylint: disable=C0116
            return getattr(self, attr_name)

        getter_name = f"get_{attr_name}"
        setattr(self.__class__, getter_name, getter)

    def _create_setter(self, attr_name):
        """
        A method which creates a setter for a given attribute
        """

        def setter(self, value):  # pylint: disable=C0116
            setattr(self, attr_name, value)

        setter_name = f"set_{attr_name}"
        setattr(self.__class__, setter_name, setter)

    def add_attribute(self, attr_name, value):
        """
        A method which adds an attribute to the class and creates a setter and a getter for it
        """
        setattr(self, attr_name, value)
        self._create_getter(attr_name)
        self._create_setter(attr_name)
