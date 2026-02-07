#  Copyright (c) 2026. The neuraLQX Authors - All Rights Reserved
#
#  Licensed under the Apache License 2.0, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


import pytest
import numpy as np


@pytest.fixture(scope="session")
def hilbert(nk):
    return nk.hilbert.Spin(s=1 / 2, N=4)


@pytest.fixture(scope="session")
def site():
    return 1


@pytest.fixture(scope="session")
def complex_single_site_matrix():
    return np.array([[0.25, 1.0 + 2.0j], [3.0 - 1.0j, -0.75]], dtype=np.complex128)
