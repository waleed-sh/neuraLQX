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
This file contains an exact sampler (NetKet implementation)
"""

from netket.sampler import ExactSampler as ExSamp


class ExactSampler(ExSamp):
    """
    This class is just a wrapper for the netket.sampler.ExactSampler class
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.__name__ = "Exact Sampler"
