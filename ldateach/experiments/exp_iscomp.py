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


from ldateach.fastlda import teach_lda_pgis
from ldateach.fastlda import teach_lda_unis
from ldateach.fastlda import evidence_lda_hm
from ldateach.utils import gen_docs
from ldateach.utils import isrelerr
from scipy.misc import logsumexp
from scipy.stats import nanmean
from multiprocessing.pool import Pool
from math import log

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import random

sns.set_context("paper")


def _expitr(docs, dtpc, alpha, beta, n_samples, seed):
    n_topics = len(dtpc)
    n_words = len(dtpc[0])

    unis = np.array(teach_lda_unis(docs, dtpc, alpha=alpha, beta=beta,
                                   n_samples=n_samples, seed=seed,
                                   return_parts=True)[-1])
    unis += sum([len(d['w']) for d in docs])*log(n_topics)

    pgis = np.array(teach_lda_pgis(docs, dtpc, alpha=alpha, beta=beta,
                                   n_samples=n_samples, seed=seed+1,
                                   return_parts=True)[-1])
    hm = evidence_lda_hm(docs, n_topics, n_words, alpha=alpha, beta=beta,
                         n_samples=n_samples, stop_itr=200, seed=seed+2,
                         return_samples=True)

    lsxaccum = lambda x: logsumexp(x) - log(len(x))
    hmaccum = lambda x: -(logsumexp(x) - log(len(x)))

    data = []
    steps = np.unique(np.array(np.exp(np.linspace(
        log(2), log(n_samples-1), 250))))
    for i in steps:
        n_samples = float(i+1)
        data.append({'Samples': n_samples,
                     'Type': 'Uniform',
                     'Estimate': lsxaccum(unis[:i]),
                     'Relative error': isrelerr(unis[:i]), })
        data.append({'Samples': n_samples,
                     'Type': 'SIS',
                     'Estimate': lsxaccum(pgis[:i]),
                     'Relative error': isrelerr(pgis[:i]), })
        data.append({'Samples': n_samples,
                     'Type': 'Harmonic mean',
                     'Estimate': hmaccum(hm[:i]),
                     'Relative error': isrelerr(hm[:i]), })
    return data


def expitr(args):
    return _expitr(*args)


def runexp(n_docs, n_words, n_words_per_doc, n_topics, alpha, beta, n_runs,
           n_samples):
    random.seed(1990)
    np.random.seed(1999)
    args = []
    docs, dtpc = gen_docs(n_docs, n_topics, n_words, n_words_per_doc,
                          alpha, beta, noise=.1)
    for i in range(n_runs):
        seed_i = random.randrange(2**31)
        args.append((docs, dtpc, alpha, beta, n_samples, seed_i,))

    pool = Pool()
    mapper = pool.map
    # mapper = lambda func, argsin: [func(a) for a in argsin]

    res = mapper(expitr, args)
    return pd.DataFrame([item for sublist in res for item in sublist])


def plotexp(df):
    plt.figure(figsize=(7.5, 7.5), tight_layout=True, dpi=150)
    plt.subplot(2, 1, 1)
    sns.tsplot(data=df, value='Estimate', time='Samples', condition='Type',
               estimator=nanmean)

    plt.gca().set_xscale("log")
    plt.xlim([0, df['Samples'].max()])

    plt.subplot(2, 1, 2)
    sns.tsplot(data=df, value='Relative error', time='Samples',
               condition='Type', estimator=nanmean)

    plt.gca().set_xscale("log")
    plt.xlim([0, df['Samples'].max()])
    plt.savefig('iscomp.png', dpi=300)


if __name__ == "__main__":
    # res = runexp(1, 20386, 112, 300, 50./300., .1, 8, 1000)
    res = runexp(1, 10, 20, 5, 50./300., .1, 8, 1000)
    plotexp(res)
