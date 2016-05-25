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


DOCSTR = """ Complexity experiment.
Time/samples to relative error, ε, for different numbers of topics, documents,
vocab words, and total words.
"""

from ldateach.fastlda import teach_lda_pgis
from ldateach.utils import gen_docs
from ldateach.utils import isrelerr
from multiprocessing.pool import Pool

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
import itertools as it
import numpy as np

import random
import pandas
import time
import pickle
import os

sns.set_context('paper')

MAX_ITR = 100000
DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(DIR, 'data')


def expitr(n_topics, n_vocab, n_words, α, β, n_docs, tol, seed, tasknum):
    random.seed(seed)

    print('Running %d topics, %d  n_vocab, and %d words to ε=%1.2f.' %
          (n_topics, n_vocab, n_words, tol,))

    docs, topics = gen_docs(n_docs, n_topics, n_vocab, n_words, α, β)

    i = 0
    t_pgis = 0
    relerr = float('Inf')
    weights = []
    while relerr > tol and i < MAX_ITR:
        i += 1
        seed = random.randrange(2**31)

        t_start = time.time()
        _, _, lml = teach_lda_pgis(docs, topics, alpha=α, beta=β, n_samples=1,
                                   seed=seed, return_parts=True)
        t_end = time.time()

        t_pgis += t_end - t_start
        weights.append(lml)

        if i > 1:
            relerr = isrelerr(weights)

    if i > MAX_ITR:
        print('Warning: MAX_ITR reached. Breaking.')

    return t_pgis, i, n_topics, n_vocab, n_words, α, β, n_docs


def _expitr(args):
    return expitr(*args)


def runexp(n_topics_lst, n_vocab_lst, n_words_lst, alpha_lst, beta_lst,
           n_docs, n_runs, tol=.05):
    kwargsin = locals()
    params = it.product(n_topics_lst, n_vocab_lst, n_words_lst, alpha_lst,
                        beta_lst)

    args = []
    for i, (n_topics, n_vocab, n_words, α, β) in enumerate(params):
        args.append([n_topics, n_vocab, n_words, α, β, n_docs, tol])

    args = [tuple(arg + [random.randrange(2**31), i]) for
            i, arg in enumerate(args*n_runs)]

    pool = Pool()
    res = pool.map(_expitr, args)

    data = []
    for t, n, n_topics, n_vocab, n_words, α, β, n_docs in res:
        datum = {'time': t,
                 'Number of samples': n,
                 'Number of topics': n_topics,
                 'Vobabulary size': n_words,
                 'Number of words': n_words,
                 'Number of documents': n_docs,
                 'α': α,
                 'β': β, }
        data.append(datum)

    res = {
        'df': pandas.DataFrame(data),
    }
    return {'res': res, 'args': kwargsin}


def plotexp(res, filename):
    parser = argparse.ArgumentParser(description=DOCSTR)
    parser.add_argument('-b', '--beta', type=float, default=.1)

    df = res['res']['df']

    # TODO: Adjust figure size for print
    sns.factorplot(data=df, y='Number of samples', x='Number of words',
                   hue='Number of topics', col='α', row='β', legend_out=True,
                   size=1.75, aspect=1, ci=95)
    # sns.factorplot(data=df, y='Number of samples', x='Number of words',
    #                hue='Number of topics', legend_out=True,
    #                size=2, aspect=1.75)

    plt.savefig(filename, dpi=300)


if __name__ == '__main__':
    import argparse

    filename = os.path.join(DATA_DIR, 'exp_complexity.pkl')

    parser = argparse.ArgumentParser(description=DOCSTR)
    parser.add_argument('-f', '--filename', type=str, default=filename,
                        help="Output filename")
    parser.add_argument('-r', '--n_runs', type=int, default=128,
                        help="Number of runs")
    parser.add_argument('-t', '--topic_step_size', type=int, default=4,
                        help="Increment size for n_topics lists")
    parser.add_argument('-w', '--word_step_size', type=int, default=5,
                        help="Increment size for n_topics lists")
    parser.add_argument('-v', '--vocab_size', type=int, default=50,
                        help="Number of words in the vocabulary.")
    parser.add_argument('-a', '--alpha_list', type=float, nargs='+',
                        default=[.1], help="values for alpha")
    parser.add_argument('-b', '--beta_list', type=float, nargs='+',
                        default=[.1], help="Values for beta")
    parser.add_argument('--min_topics', type=int, default=2,
                        help='Minimum number of topics')
    parser.add_argument('--max_topics', type=int, default=40,
                        help='Maximum number of topics')
    parser.add_argument('--min_words', type=int, default=5,
                        help='Minimum number of words')
    parser.add_argument('--max_words', type=int, default=60,
                        help='Maximum number of words')
    parser.add_argument('--plot_only', action='store_true', default=False,
                        help='Plot only; do not run experiment.')

    kwargs = vars(parser.parse_args())

    # ---
    n_topics_lst = np.arange(kwargs['min_topics'], kwargs['max_topics'],
                             kwargs['topic_step_size'])
    n_words_lst = np.arange(kwargs['min_words'], kwargs['max_words'],
                            kwargs['word_step_size'])
    n_vocab_lst = [kwargs['vocab_size']]
    n_docs = 1

    if not kwargs['plot_only']:
        res = runexp(n_topics_lst, n_vocab_lst, n_words_lst,
                     kwargs['alpha_list'], kwargs['beta_list'], n_docs,
                     kwargs['n_runs'], tol=.05)
        pickle.dump(res, open(kwargs['filename'], 'wb'))

    res = pickle.load(open(kwargs['filename'], 'rb'))
    plotexp(res, filename + '.png')
