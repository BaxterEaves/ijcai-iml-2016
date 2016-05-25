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
import random
import numpy as np

from ldateach.fastlda import teach_lda_pgis as pgis
from ldateach.fastlda import teach_lda_unis as unis
from ldateach.fastlda import teach_lda_exact

from scipy.stats import linregress
from scipy.misc import logsumexp
from ldateach.utils import gen_docs


ALPHA = .1
BETA = .8


def docset(seed):
    np.random.seed(seed)
    random.seed(seed)

    docs, tpcs = gen_docs(n_docs=2, n_topics=2, n_words=10, n_words_per_doc=6)
    return docs, tpcs


def logcumexp(pr, like, marg):
    assert len(like) == len(marg)
    ans = []
    for i in range(10, len(like), 2):
        ans.append(pr + logsumexp(like[:i]) - logsumexp(marg[:i]))
    return np.array(ans)


# -- Make sure the estimates approach the correct answer
@pytest.mark.parametrize('seed,estimator', [(1337, unis,), (1337, pgis,),
                                            (128, unis,), (128, pgis,)])
def test_pgis(seed, estimator):
    docs, tpcs = docset(seed)

    pt = teach_lda_exact(docs, tpcs, alpha=ALPHA, beta=BETA)

    errs = []
    for i in range(10):
        sd = random.randrange(2**31)
        pr, like, marg = estimator(docs, tpcs, alpha=ALPHA, beta=BETA,
                                   seed=sd, return_parts=True, n_samples=1500)

        err = np.abs(pt-logcumexp(pr, like, marg))
        errs.append(err)

    err = np.mean(errs, axis=0)
    slope, _, _, p, _ = linregress(np.arange(len(err)-100), err[100:])
    assert slope < 0. and p < .05

    vars = np.std(errs, axis=0)
    slope, _, _, p, _ = linregress(np.arange(len(vars)), vars)
    assert slope < 0. and p < .05
