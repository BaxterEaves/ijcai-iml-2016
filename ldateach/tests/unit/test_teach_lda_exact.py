# ldateach, generate/choose documents to teach topics model
# Copyright (C) 2016 Baxter S. Eaves Jr.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


import pytest

from scipy.misc import logsumexp
from ldateach.fastlda import teach_lda_exact

FLOAT_ERROR_TOL = 10E-6


@pytest.fixture
def docset():
    docs = [{'w': [0, 1]}, {'w': [2, 0]}]
    topics = [[.8, .1, .1], [.2, .3, .5]]
    alpha = .5
    beta = .25
    return docs, topics, alpha, beta


def test_valid_output(docset):
    docs, topics, _, _ = docset
    val = teach_lda_exact(docs, topics)

    assert isinstance(val, float)


def test_valid_parts_output(docset):
    docs, topics, _, _ = docset
    vals = teach_lda_exact(docs, topics, return_parts=True)

    assert len(vals) == 3

    prior, numer, denom = vals

    assert isinstance(prior, float)
    assert isinstance(numer, list)
    assert isinstance(denom, list)
    # should be an entry for each assignment
    assert len(numer) == 2**4
    assert len(denom) == 2**4


def test_values_all_topics(docset):
    docs, topics, alpha, beta = docset
    val = teach_lda_exact(docs, topics, alpha=alpha, beta=beta)

    assert abs(val - 0.1231028690919338) < FLOAT_ERROR_TOL

    r, l, m = teach_lda_exact(docs, topics, alpha=alpha, beta=beta,
                              return_parts=True)

    assert abs(r - (-1.0704195193591608)) < FLOAT_ERROR_TOL
    assert abs(logsumexp(l) - (-4.5853675586919111)) < FLOAT_ERROR_TOL
    assert abs(logsumexp(m) - (-5.7788899471430053)) < FLOAT_ERROR_TOL


def test_values_single_topic_topics(docset):
    docs, topics, alpha, beta = docset
    val = teach_lda_exact(docs, topics, [0], alpha=alpha, beta=beta)

    assert abs(val - 0.5515582903356044) < FLOAT_ERROR_TOL

    r, l, m = teach_lda_exact(docs, topics, [0], alpha=alpha, beta=beta,
                              return_parts=True)

    assert abs(r - (-0.0395513196862107)) < FLOAT_ERROR_TOL
    assert abs(logsumexp(l) - (-5.1877803371211897)) < FLOAT_ERROR_TOL
    assert abs(logsumexp(m) - (-5.7788899471430053)) < FLOAT_ERROR_TOL
