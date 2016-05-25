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

from ldateach.fastlda import FastLDA
from ldateach.fastlda import FastLazyLDA
from ldateach.fastlda import FastSparseLDA
from ldateach.fastlda import FastLightLDA
from ldateach.lda import LDA
from ldateach.lda import LazyLDA
from ldateach.lda import SparseLDA
from ldateach.utils import discrete_draw
from scipy.stats import dirichlet
from sklearn.metrics import adjusted_rand_score as ari
from scipy.stats import ks_2samp
from math import log

import random
import numpy as np


def _geweke_draw_docs(docs, asgn, n_dk, n_kw, alpha, beta, init=False):
    n_dk = np.array(n_dk)
    n_kw = np.array(n_kw)

    n_topics = len(n_kw)
    n_words = len(n_kw[0])

    n_d = np.sum(n_dk, axis=1)
    assert len(n_d) == len(docs)

    for d, doc in enumerate(docs):
        for w, wrd in enumerate(doc['w']):
            tpc = asgn[d][w]

            n_dk[d, tpc] -= 1
            n_d[d] -= 1
            if not init:
                n_kw[tpc, wrd] -= 1

            logps = np.zeros(n_words)
            for widx in range(n_words):
                logp = log(n_kw[tpc, widx] + beta)
                logp += log(n_dk[d, tpc] + alpha)
                logp -= log(n_d[d] + alpha*n_topics)
                logps[widx] = logp

            new_wrd = discrete_draw(logps, logp=True)
            docs[d]['w'][w] = new_wrd

            n_dk[d, tpc] += 1
            n_kw[tpc, new_wrd] += 1
            n_d[d] += 1

    assert np.sum(n_dk) == np.sum(n_kw)
    assert np.sum(n_d) == np.sum(n_dk)

    phi = n_kw + beta
    phi /= np.sum(phi, axis=1)[:, np.newaxis]

    return docs, phi


def _get_lda_properties(lda):
    z = lda.get_radix()
    asgn = lda.get_assignment()
    n_dk = lda.n_dk
    n_kw = lda.n_kw

    return z, asgn, n_dk, n_kw


def _insert_stats(stats_dict, docs, d_ref, z_ref, z_lda, phi, idx):
    φ = np.array(phi).flatten()

    docsf = np.array([doc['w'] for doc in docs]).flatten()

    stats_dict['doc_ari'][idx] = ari(d_ref, docsf)
    stats_dict['ari'][idx] = ari(z_ref, z_lda)
    stats_dict['phi_std'][idx] = np.std(φ)

    return stats_dict


def _forward_sample(ldactr, n_docs, n_topics, n_words, n_words_per_doc, alpha,
                    beta):
    # docs, _ = gen_docs(n_docs, n_topics, n_words, n_words_per_doc, alpha, beta)
    # lda = ldactr(docs, n_topics, n_words, alpha, beta, seed)
    # z, asgn, n_dk, _ = _get_lda_properties(lda)
    docs = [{'w': [0]*n_words_per_doc} for _ in range(n_docs)]
    n_dk = np.zeros((n_docs, n_topics,))
    n_kw = np.zeros((n_topics, n_words,))
    asgn = []
    for d in range(n_docs):
        asgn_d = []
        theta_d = dirichlet.rvs([alpha]*n_topics)[0]
        for _ in range(n_words_per_doc):
            k = discrete_draw(theta_d)
            asgn_d.append(k)
            n_dk[d, k] += 1
        asgn.append(asgn_d)
    z = [item for sublist in asgn for item in sublist]
    docs, phi = _geweke_draw_docs(docs, asgn, n_dk, n_kw, alpha, beta,
                                  init=True)
    return docs, z, phi


def _transition_sample(ldactr, lda, docs, n_topics, n_words, alpha, beta, seed):
    z = lda.get_assignment()
    lda = ldactr(docs, n_topics, n_words, alpha, beta, seed=seed)
    lda.set_assignment(z)
    lda.run(1)
    z, asgn, n_dk, n_kw = _get_lda_properties(lda)
    docs, phi = _geweke_draw_docs(docs, asgn, n_dk, n_kw, alpha, beta)
    return lda, docs, z, phi


# ---
def geweke_test_lda(ldactr, n_samples, n_chains, n_docs=3, n_topics=3,
                    n_words=10, n_words_per_doc=20, alpha=1., beta=1.,
                    seed=1337):
    np.random.seed(seed)
    random.seed(seed)

    z_rnd = np.random.randint(n_topics, size=n_docs*n_words_per_doc)
    d_rnd = []
    for _ in range(n_docs):
        d_rnd.append(np.random.randint(n_words, size=n_words_per_doc))
    d_rnd = np.array(d_rnd).flatten()

    # forward samples
    stats_f = {
        'ari': np.zeros(n_samples),
        'doc_ari': np.zeros(n_samples),
        'phi_std': np.zeros(n_samples),
    }
    for i in range(n_samples):
        seed = random.randrange(2**31)
        docs, z, phi = _forward_sample(ldactr, n_docs, n_topics, n_words,
                                       n_words_per_doc, alpha, beta)
        stats_f = _insert_stats(stats_f, docs, d_rnd, z_rnd, z, phi, i)

    # samples from posterior chain
    stats_p = {
        'ari': np.zeros(n_samples),
        'doc_ari': np.zeros(n_samples),
        'phi_std': np.zeros(n_samples),
    }
    i = 0
    for _ in range(n_chains):
        seed = random.randrange(2**31)
        docs, z, phi = _forward_sample(ldactr, n_docs, n_topics, n_words,
                                       n_words_per_doc, alpha, beta)
        lda = ldactr(docs, n_topics, n_words, alpha, beta, seed=seed)
        asgn = [z[n_words_per_doc*c:n_words_per_doc*(c+1)]
                for c in range(n_docs)]
        lda.set_assignment(asgn)

        for _ in range(int(n_samples/n_chains)):
            seed = random.randrange(2**31)
            lda, docs, z, phi = _transition_sample(
                ldactr, lda, docs, n_topics, n_words, alpha, beta, seed)
            stats_p = _insert_stats(stats_p, docs, d_rnd, z_rnd, z, phi, i)
            i += 1

    return stats_f, stats_p


# ---
@pytest.mark.parametrize('ldacnstr', [LDA, FastLDA, LazyLDA, SparseLDA,
                                      FastLazyLDA, FastSparseLDA])
def test_lda_geweke(ldacnstr):
    stats_f, stats_p = geweke_test_lda(ldacnstr, 2000, 1)
    statkeys = [k for k in stats_f.keys()]
    for i, stat in enumerate(statkeys):
        sf = stats_f[stat]
        sp = stats_p[stat]
        _, p = ks_2samp(sf, sp)
        assert p > .1


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns
    from ldateach.plot_utils import pp_plot
    # from statsmodels.graphics.gofplots import qqplot_2samples

    sns.set_context("notebook")
    stats_f, stats_p = geweke_test_lda(FastLightLDA, 10000, 1)

    statkeys = [k for k in stats_f.keys()]
    plt.figure(tight_layout=True)
    for i, stat in enumerate(statkeys):
        sf = stats_f[stat]
        sp = stats_p[stat]

        ax = plt.subplot(1, len(statkeys), i+1)
        m = np.mean(sf)
        s = np.std(sf)
        _, p = ks_2samp(sf, sp)
        print("KS-Test for %s: p=%f" % (stat, p,))

        pp_plot(sf, sp, nbins=200)
        plt.xlabel(stat + " (FORWARD)")
        plt.ylabel(stat + " (POSTERIOR)")

    plt.show()
