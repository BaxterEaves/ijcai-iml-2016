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


DOCSTR = """ Generate documents from the teaching model and randomly from LDA
and compare performance of LDA at recovering the topics.
"""

from ldateach.fastlda import teach_lda_pgis
from ldateach.utils import gen_docs
from ldateach import fastlda
from ldateach import utils

from math import log
from multiprocessing.pool import Pool
import pandas as pd
import numpy as np
import pickle as pkl
import random
import copy
import sys
import os

import matplotlib as mpl
mpl.use('Agg')

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context('paper')
DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(DIR, 'data')


def _get_err(docs, topics, n_words, alpha, beta, n_steps, seed):
    """ FIXME """
    n_topics = len(topics)
    lda = fastlda.FastLDA(docs, n_topics, n_words, alpha, beta, seed)
    lda.run(n_steps)
    return utils.lda_topic_err(topics, lda.n_kw, alpha)


def _jitter_docs(docsin, ρ, n_wrds):
    """ Replaces a existing word with a uniform random word in a document with
    probability ρ. Does not completely update the data structure."""
    docs = copy.deepcopy(docsin)
    for i in range(len(docs)):
        for j in range(len(docs[i]['w'])):
            if random.random() < ρ:
                docs[i]['w'][j] = random.randrange(n_wrds)
    return docs


def _do_chain(args):
    """
    Generate a synthetic documen set from the teaching model.

    Return the documents and the acceptance rate of the sampler
    """
    docs_r = args[0]
    docs_t = args[1]  # start state for teaching
    topics = args[2]  # target topics

    α = args[3]
    β = args[4]
    ρ = args[5]
    n_samples = args[6]
    n_iter = args[7]
    seed = args[8]
    chain = args[9]

    n_lda_steps = 500

    n_words = len(topics[0])
    docs_t = _jitter_docs(docs_t, 1.0, n_words)

    n_acpt = 0
    acr = 0

    lp = teach_lda_pgis(docs_t, topics, alpha=α, beta=β, n_samples=n_samples)
    # lp = teach_lda_exact(docs_t, topics, alpha=α, beta=β)

    for i in range(n_iter):
        docs_p = _jitter_docs(docs_t, ρ, n_words)
        lp_p = teach_lda_pgis(docs_p, topics, alpha=α, beta=β,
                              n_samples=n_samples)

        # lp_p = teach_lda_exact(docs_p, topics, alpha=α, beta=β)
        if log(random.random()) < lp_p - lp:
            n_acpt += 1
            docs_t = docs_p
            lp = lp_p

        acr = n_acpt/float(i+1)

    err_t = _get_err(docs_t, topics, n_words, α, β, n_lda_steps, seed)
    err_r = _get_err(docs_r, topics, n_words, α, β, n_lda_steps, seed)

    print(".", end="")
    sys.stdout.flush()
    return docs_t, acr, err_t, err_r, chain


# ---
def runexp(n_docs_lst=(3,), n_topics=3, n_words=10, n_words_per_doc=12,
           alpha=.8, beta=.8, n_samples=20000, n_iter=5000, n_runs=500,
           seed=None, filename='synthetic.pkl'):

    # tuning the MH variance parameter.
    ρ = .15  # proportion of words to randomly flip

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    pool = Pool()

    # generate starting docs
    res = {}
    for n_docs in n_docs_lst:
        print("*Running %d documents." % (n_docs,))
        res[n_docs] = {}
        args = []
        for chain in range(n_runs):
            seed = random.randrange(2**31)
            docs_r, topics = gen_docs(n_docs, n_topics, n_words,
                                      n_words_per_doc, alpha, beta)
            docs_t = _jitter_docs(docs_r, 1.0, n_words)
            args.append((docs_r, docs_t, topics, alpha, beta, ρ, n_samples,
                         n_iter, seed, chain))
            res[n_docs][chain] = {
                'docs_r': docs_t,
                'topics': topics,
                'args': args[-1]}

        out = pool.map(_do_chain, args)
        print("")

        for docs_t, acr, err_t, err_r, idx in out:
            res[n_docs][idx]['docs_t'] = docs_t
            res[n_docs][idx]['acr'] = acr
            res[n_docs][idx]['err_t'] = err_t
            res[n_docs][idx]['err_r'] = err_r

    pkl.dump(res, open(filename, 'wb'))


def plotexp(filename):
    res = pkl.load(open(filename, 'rb'))

    dlist = []
    for n_docs, res_d in res.items():
        for i, out in enumerate(res_d.values()):
            err_t = out['err_t']
            err_r = out['err_r']
            dlist.append({'n_docs': n_docs, 'error': err_t, 'type': 'teaching'})
            dlist.append({'n_docs': n_docs, 'error': err_r, 'type': 'random'})

    data = pd.DataFrame(dlist)

    plt.figure(figsize=(7.5, 3.5), facecolor='white', tight_layout=True)
    sns.violinplot(data=data, x='n_docs', y='error', hue='type', split=True,
                   inner='quartile', palette='Set1')
    plt.ylabel('Sum squared topic error')
    plt.xlabel('Number of documents')

    plt.savefig(filename + '.png', dpi=300)


if __name__ == "__main__":
    import argparse

    filename = os.path.join(DATA_DIR, 'synthetic.pkl')

    parser = argparse.ArgumentParser(description=DOCSTR)
    parser.add_argument('-t', '--n_topics', type=int, default=3)
    parser.add_argument('-w', '--n_words_per_doc', type=int, default=10)
    parser.add_argument('-s', '--n_samples', type=int, default=5000)
    parser.add_argument('-r', '--n_runs', type=int, default=128)
    parser.add_argument('-a', '--alpha', type=float, default=.1)
    parser.add_argument('-b', '--beta', type=float, default=.1)
    parser.add_argument('-f', '--filename', type=str, default=filename)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--plot_only', action='store_true', default=False,
                        help='Plot only; do not run experiment.')
    parser.add_argument('-d', '--n_docs_lst', type=int, nargs='+',
                        default=[2], help='Number of documents')
    parser.add_argument('-v', '--n_words', type=int, default=5,
                        help='Number of words in the vocabular.')
    parser.add_argument('-i', '--n_iter', type=int, default=1000,
                        help='Number of Metrpolis transition')

    kwargs = vars(parser.parse_args())

    if not kwargs['plot_only']:
        del kwargs['plot_only']
        runexp(**kwargs)

    plotexp(kwargs['filename'])
