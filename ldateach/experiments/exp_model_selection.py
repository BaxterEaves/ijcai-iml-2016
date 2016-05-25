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


DOCSTR = """ Plots the evidence by number of topics for a corpus.
Uses the same procedure as in "Finding scientific topics". β is set to 0.1 and
α is set to 50/T, where T is the number of topics.
"""

from ldateach.fastlda import teach_lda_pgis
from ldateach.fastlda import evidence_lda_hm
from ldateach.utils import loglinspace
from multiprocessing.pool import Pool
from scipy.misc import logsumexp
from math import log
from itertools import product

import matplotlib as mpl
mpl.use('Agg')

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import time
import pickle
import random
import os

sns.set_context('paper')
sns.set_palette('gray')

DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(DIR, 'data')
SCRAPE_DIR = os.path.join(DIR, '..', 'scrape', 'data')
USE_HARMONIC_MEAN = False


def get_imdb_docs(filename):
    metadata = pickle.load(open(filename, 'rb'))
    docs = metadata['docs']
    words = metadata['words']

    n_words = len(words)

    return docs, n_words


def evidence(docs, n_topics, n_words, beta=.1, n_samples=10000, seed=1337):
    """ estimate the evidence using sequental importance sampling """

    # generate fake topics. Remember, teach_lda_XXX calcualtes the teaching
    # probability; we're onlt concerned with the evidence
    # TODO: Pass m as an argument
    m = 50.
    alpha = m/n_topics
    if USE_HARMONIC_MEAN:
        evidence = evidence_lda_hm(docs, n_topics, n_words, alpha=alpha,
                                   beta=beta, n_samples=n_samples,
                                   stop_itr=1000, seed=None)
    else:
        topics = np.ones((n_topics, n_words))/n_words
        _, _, evx = teach_lda_pgis(docs, topics, alpha=alpha, beta=beta,
                                   n_samples=n_samples, seed=seed,
                                   return_parts=True)

        # the evidence isn't normalized in pgis because the normalizer cancels
        # out of the ratio. Normalize here.
        evidence = logsumexp(evx) - log(n_samples)
    return evidence


def expitr(docs, n_topics, n_words, beta, n_samples, seed, threadid):
    print("Calculating evidence for %d topics. (THREAD: %d)" %
          (n_topics, threadid,))
    t_start = time.time()
    evx = evidence(docs, n_topics, n_words, beta, n_samples, seed)
    runtime = time.time() - t_start
    return evx, runtime, n_topics


def _expitr(args):
    return expitr(*args)


def runexp(filename, n_topics_range, β=.1, n_samples=1000, n_runs=8,
           seed=1337):
    kwargsin = locals()
    docs, n_words = get_imdb_docs(filename)
    random.seed(seed)

    args = []
    pool = Pool()
    for tid, (_, n_topics) in enumerate(product(range(n_runs), n_topics_range)):
        seed_i = random.randrange(2**31)
        args.append((docs, n_topics, n_words, β, n_samples, seed_i, tid,))

    data = []
    runtime = []
    res = pool.map(_expitr, args)
    for evx, rt, nt in res:
        data.append({'Log evidence': evx,
                     'Runtime': rt,
                     'Number of topics': nt})

    out = {
        'res': {
            'data': pd.DataFrame(data),
        },
        'args': kwargsin,
    }
    return out


def plotexp(res):
    data = res['res']['data']
    _, basename = os.path.split(res['args']['filename'])
    basename = basename.replace('.', '')

    plt.figure(facecolor='white', tight_layout=True, figsize=(4.5, 3.5),
               dpi=300)
    sns.pointplot(x='Number of topics', y='Log evidence', data=data)
    plt.savefig(os.path.join(DATA_DIR, basename + '_nt_evidence.png'), dpi=300)

    plt.figure(facecolor='white', tight_layout=True, figsize=(4.5, 3.5),
               dpi=300)
    sns.pointplot(x='Number of topics', y='Runtime', data=data)
    plt.savefig(os.path.join(DATA_DIR, basename + '_nt_runtime.png'), dpi=300)


if __name__ == "__main__":
    import argparse

    corpus_filename = os.path.join(SCRAPE_DIR, 'imdb', 'imdb1k.csv.pkl')
    res_filename = os.path.join(DATA_DIR, 'n_topics_selection.pkl')

    parser = argparse.ArgumentParser(description=DOCSTR)
    parser.add_argument('-b', '--beta', type=float, default=.1)
    parser.add_argument('-s', '--n_samples', type=int, default=1000)
    parser.add_argument('-w', '--n_words', type=int, default=5)
    parser.add_argument('-m', '--min_topics', type=int, default=2)
    parser.add_argument('-x', '--max_topics', type=int, default=100)
    parser.add_argument('-n', '--n_steps', type=int, default=20)
    parser.add_argument('-r', '--n_runs', type=int, default=8)
    parser.add_argument('--corpus_filename', type=str, default=corpus_filename,
                        help="path to processed corpus data file")
    parser.add_argument('--res_filename', type=str, default=res_filename,
                        help="save file name")

    kwargs = vars(parser.parse_args())

    n_topics_range = loglinspace(kwargs['min_topics'], kwargs['max_topics'],
                                 kwargs['n_steps'])

    res = runexp(kwargs['corpus_filename'], n_topics_range, kwargs['beta'],
                 kwargs['n_samples'], kwargs['n_runs'], seed=1337)
    pickle.dump(res, open(kwargs['res_filename'], 'wb'))

    res = pickle.load(open(kwargs['res_filename'], 'rb'))
    plotexp(res)
