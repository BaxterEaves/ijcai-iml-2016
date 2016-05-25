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
from ldateach.fastlda import teach_lda_exact
from ldateach.fastlda import evidence_lda_hm
from ldateach.utils import gen_docs
from scipy.misc import logsumexp
from multiprocessing.pool import Pool
from math import log

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random

sns.set_context("paper")


def _expitr(docs, dtpc, alpha, beta, n_samples, seed):
    logsamples = log(n_samples)
    n_topics = len(dtpc)
    n_words = len(dtpc[0])

    exact = logsumexp(teach_lda_exact(docs, dtpc, alpha=alpha, beta=beta,
                                      return_parts=True)[-1])
    print("1")
    unis = logsumexp(teach_lda_unis(docs, dtpc, alpha=alpha, beta=beta,
                                    n_samples=n_samples, seed=seed,
                                    return_parts=True)[-1]) - logsamples
    print("2")
    unis += sum(len(d['w']) for d in docs)*log(n_topics)
    pgis = logsumexp(teach_lda_pgis(docs, dtpc, alpha=alpha, beta=beta,
                                    n_samples=n_samples, seed=seed+1,
                                    return_parts=True)[-1]) - logsamples
    print("3")
    hm = evidence_lda_hm(docs, n_topics, n_words, alpha=alpha, beta=beta,
                         n_samples=n_samples, stop_itr=100, seed=seed+2)
    print("4")
    print(seed)

    return {'Exact': exact, 'Uniform': unis, 'SIS': pgis, 'HM': hm}


def expitr(args):
    return _expitr(*args)


def runexp(n_docs, n_words, n_words_per_doc, n_topics, alpha, beta, n_runs,
           n_samples):
    random.seed(1337)
    args = []
    for i in range(n_runs):
        seed_i = random.randrange(2**31)
        docs, dtpc = gen_docs(n_docs, n_topics, n_words, n_words_per_doc,
                              alpha, beta, noise=.1)
        args.append((docs, dtpc, alpha, beta, n_samples, seed_i,))

    pool = Pool()
    mapper = pool.map
    # mapper = lambda func, argsin: [func(a) for a in argsin]

    res = mapper(expitr, args)
    return pd.DataFrame(res)


def plotexp(df):

    minexact = df['Exact'].min()
    maxexact = df['Exact'].max()

    el = [minexact, maxexact]

    plt.figure(figsize=(7.5, 7.5), tight_layout=True)

    plt.subplot(2, 2, 1)
    plt.scatter(df['Exact'].values, df['Uniform'].values, color='black')
    plt.plot(el, el, color='red', alpha=.8)
    plt.xlabel('Exact')
    plt.ylabel('Uniform estimate')

    plt.subplot(2, 2, 2)
    plt.scatter(df['Exact'].values, df['SIS'].values, color='black')
    plt.plot(el, el, color='red', alpha=.8)
    plt.xlabel('Exact')
    plt.ylabel('SIS estimate')

    plt.subplot(2, 2, 3)
    plt.scatter(df['Exact'].values, df['HM'].values, color='black')
    plt.plot(el, el, color='red', alpha=.8)
    plt.xlabel('Exact')
    plt.ylabel('Harmonic mean estimate')

    plt.subplot(2, 2, 4)
    sns.kdeplot(df['Uniform']-df['Exact'], shade=True, label='Uniform')
    sns.kdeplot(df['SIS']-df['Exact'], shade=True, label='SIS')
    sns.kdeplot(df['HM']-df['Exact'], shade=True, label='Harmonic mean')
    plt.xlabel('Estimate - Exact')

    plt.show()


if __name__ == "__main__":
    res = runexp(2, 5, 4, 2, 50/3., .1, 100, 1000)

    plotexp(res)
