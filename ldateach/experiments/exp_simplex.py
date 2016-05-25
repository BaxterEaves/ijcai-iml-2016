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


DOCSTR = """ Plot teaching distribution on simplex.
Compares the teaching distribution to the likelihood by plotting the both on
a simplex for a small problem.
"""

from ldateach.fastlda import teach_lda_exact

from multiprocessing.pool import Pool
from scipy.misc import logsumexp
from math import pi, cos, sin
import itertools as it
import numpy as np
import pickle as pkl
import random
import os

import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
from ldateach.plot_utils import MidpointNorm
import matplotlib.tri as tri
import seaborn as sns

sns.set_context('paper')

DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(DIR, 'data')

_corners = np.array([[0, 0], [1, 0], [0.5, 0.75**0.5]])
_triangle = tri.Triangulation(_corners[:, 0], _corners[:, 1])
_midpoints = [(_corners[(i + 1) % 3] + _corners[(i + 2) % 3]) / 2.0
              for i in range(3)]


def _expiter_inner(docs, topics, α, β):
    """
    Calculate logp under teaching model, teaching only the first topic.
    """
    pr, nu, dn = teach_lda_exact(docs, topics, alpha=α, beta=β,
                                 return_parts=True)
    numer = logsumexp(nu)
    denom = logsumexp(dn)
    lp = pr + numer - denom

    return docs, lp, numer, denom


def _expiter(args):
    return _expiter_inner(*args[:-1]), args[-1]


def counts_to_doc(counts):
    counts = np.round(counts)
    doc = {'words': set(), 'w': [], 'counts': {}}
    for w, ct in enumerate(counts):
        if ct > 0:
            doc['words'].add(w)
            doc['counts'][w] = int(ct)
            doc['w'].extend([w]*int(ct))
    return doc


def simp2cart(simp):
    assert len(simp) == 3
    return simp.dot(_corners)


def part_to_doc(parts, n_words_per_doc):
    w = []
    ct = np.zeros(len(parts)+1)

    widx = 0
    prt = parts[0]
    pidx = 0

    for i in range(len(parts)+n_words_per_doc):
        if i == prt:
            widx += 1
            pidx += 1
            if pidx == len(parts):
                prt = n_words_per_doc*2
            else:
                prt = parts[pidx]
        else:
            w.append(widx)
            ct[widx] += 1

    assert len(w) == n_words_per_doc

    return w, ct


# ---
def runexp(n_docs=1, n_words_per_doc=10, alpha=.2, beta=.2, seed=1337,
           rotate=False, n_steps=5, filename='simplex.pkl'):
    argsin = locals()

    n_words = 3

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    def gen_all_docs(n_words_per_doc):
        docs = []
        for w in it.product(range(n_words), repeat=n_words_per_doc):
            doc = {
                'w': list(w),
                'words': set(w),
                'counts': np.bincount(list(w), minlength=3),
            }
            docs.append(doc)
        return docs

    def gen_all_docsets(n_docs, n_words_per_doc):
        """ Generate documents with known topics """
        docs_lst = gen_all_docs(n_words_per_doc)
        docsets = []
        for docset in it.product(docs_lst, repeat=n_docs):
            docsets.append(list(docset))

        return docsets

    if rotate:
        weights = np.linspace(0, pi, n_steps)

        def xy2bc(xy, tol=10E-3):
            s = [(_corners[i] - _midpoints[i]).dot(xy - _midpoints[i]) / 0.75
                 for i in range(3)]
            return np.clip(s, tol, 1.0 - tol)

        topic_0 = xy2bc(np.array([.5, .866*.25]))
        topic_1 = xy2bc(np.array([.5, .866*.6]))

        orig = (topic_0.dot(_corners) + topic_1.dot(_corners))/2.0

        def rotsimplex(simp, θ):
            rotmat = np.array([[cos(θ), -sin(θ)], [sin(θ), cos(θ)]])
            xy = simp.dot(_corners)
            xy -= orig
            xy = np.dot(xy, rotmat) + orig
            return xy2bc(xy)
    else:
        topic_0 = np.array([.8, .1, .1])
        topic_1 = np.array([.1, .1, .8])
        weights = np.linspace(.5, 1, n_steps)[::-1]

    pvals = {}
    nvals = {}
    mvals = {}
    res = {}

    pool = Pool()

    topics = {}
    for weight in weights:
        if not rotate:
            tpcs_w = [topic_0*weight + topic_1*(1-weight),
                      topic_1*weight + topic_0*(1-weight)]
        else:
            tpcs_w = [rotsimplex(tpc, weight) for tpc in [topic_0, topic_1]]
        topics[weight] = tpcs_w

        print(tpcs_w)
        args = []
        all_docs = gen_all_docsets(n_docs, n_words_per_doc)
        for didx, docs in enumerate(all_docs):
            args.append((docs, tpcs_w, alpha, beta,  didx))

        out = pool.map(_expiter, args)
        print("")

        pts = []
        res[weight] = {}
        pvals[weight] = []
        nvals[weight] = []
        mvals[weight] = []
        for (docs, logp, logl, logm), idx in out:
            pt = np.zeros(3)
            for doc in docs:
                pt += doc['counts']
            assert np.sum(pt) == n_words_per_doc*n_docs
            pt /= n_words_per_doc*n_docs

            res[weight][idx] = {}
            res[weight][idx]['docs'] = docs
            res[weight][idx]['logp'] = logp
            res[weight][idx]['logl'] = logl
            res[weight][idx]['logm'] = logm
            pts.append(simp2cart(pt))
            pvals[weight].append(logp)
            nvals[weight].append(logl)
            mvals[weight].append(logm)

        pts = np.array(pts)

    output = {
        'args': argsin,
        'topics': topics,
        'result': res,
        'pvals': pvals,
        'nvals': nvals,
        'mvals': mvals,
        'pts': pts,
    }
    pkl.dump(output, open(filename, 'wb'))


def odoc(topics, alpha, beta):
    tary = np.array(topics)
    t = .25*(np.array([1, 0, 0]) + 1/3. + np.sum(2*tary, axis=0))
    t /= np.sum(t)
    # import pdb; pdb.set_trace()
    return t


def plotexp(filename):
    results = pkl.load(open(filename, 'rb'))

    alpha = results['args']['alpha']
    beta = results['args']['beta']

    plt.figure(figsize=(7.5, 4.5), tight_layout=True)

    weights = sorted(results['pvals'].keys())
    nplts = len(weights)

    for pt, weight in enumerate(weights):
        vals = {}
        for simp, nval, pval, mval in zip(results['pts'],
                                          results['nvals'][weight],
                                          results['pvals'][weight],
                                          results['mvals'][weight]):
            simp_t = tuple(simp)
            if simp_t not in vals:
                vals[simp_t] = {'pvals': [], 'nvals': [], 'mvals': []}
            vals[simp_t]['nvals'].append(nval)
            vals[simp_t]['pvals'].append(pval)
            vals[simp_t]['mvals'].append(mval)

        pts = []
        pvals = []
        nvals = []
        mvals = []
        for simp_t in vals.keys():
            pts.append(np.array(simp_t))
            assert len(vals[simp_t]['pvals']) == len(vals[simp_t]['nvals'])
            assert len(vals[simp_t]['pvals']) == len(vals[simp_t]['mvals'])
            logn = np.log(len(vals[simp_t]['pvals']))
            pvals.append(logsumexp(vals[simp_t]['pvals']) - logn)
            nvals.append(logsumexp(vals[simp_t]['nvals']) - logn)
            mvals.append(logsumexp(vals[simp_t]['mvals']) - logn)

        pts = np.array(pts)
        x = pts[:, 0]
        y = pts[:, 1]

        pvals = np.array(pvals)
        nvals = np.array(nvals)
        mvals = np.array(mvals)

        pvals -= logsumexp(pvals)
        nvals -= logsumexp(nvals)
        mvals -= logsumexp(mvals)

        print(pvals[:10])

        # -- teaching
        ax = plt.subplot(4, nplts, pt+1)

        plt.tricontourf(x, y, np.exp(pvals), 200, cmap='Greys')
        plt.axis('equal')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 0.75**0.5])
        plt.axis('off')

        plt.hold(1)
        plt.triplot(_triangle, linewidth=.5, color='black')

        txy = np.array([simp2cart(tpc) for tpc in results['topics'][weight]])
        oxy = np.array(simp2cart(odoc(results['topics'][weight], alpha, beta)))

        ox = [oxy[0]]
        oy = [oxy[1]]

        tx = txy[:, 0]
        ty = txy[:, 1]
        plt.scatter(tx, ty, marker='+', lw=1, edgecolor='white', s=49)
        plt.scatter(ox, oy, marker='x', lw=1, edgecolor='white', s=49)

        # -- maginal
        ax = plt.subplot(4, nplts, nplts+pt+1)

        plt.tricontourf(x, y, mvals, 200, cmap='Greys')
        plt.axis('equal')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 0.75**0.5])
        plt.axis('off')

        plt.hold(1)
        plt.triplot(_triangle, linewidth=.5, color='black')
        plt.scatter(tx, ty, marker='+', lw=1, edgecolor='gray', s=49)

        # -- likelihood
        ax = plt.subplot(4, nplts, 2*nplts+pt+1)

        plt.tricontourf(x, y, nvals, 200, cmap='Greys')
        plt.axis('equal')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 0.75**0.5])
        plt.axis('off')

        plt.hold(1)
        plt.triplot(_triangle, linewidth=.5, color='black')
        plt.scatter(tx, ty, marker='+', lw=1, edgecolor='gray', s=49)

        # -- difference
        ax = plt.subplot(4, nplts, 3*nplts+pt+1)
        diff = np.exp(pvals)-np.exp(nvals)
        norm = MidpointNorm(0, vmin=min(diff), vmax=max(diff))

        plt.tricontourf(x, y, diff, 200, cmap='RdBu_r', norm=norm)
        plt.axis('equal')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 0.75**0.5])
        plt.axis('off')

        plt.hold(1)
        plt.triplot(_triangle, linewidth=.5, color='black')
        plt.scatter(tx, ty, marker='+', lw=1, edgecolor='white', s=49)

    plt.figtext(0.03, 0.88, 'Teaching', size='x-small', rotation=90)
    plt.figtext(0.03, 0.58, 'Likelihood', size='x-small', rotation=90)
    plt.figtext(0.03, 0.25, 'Difference', size='x-small', rotation=90)
    plt.savefig(filename + '.png', dpi=600)


if __name__ == "__main__":
    import argparse

    filename = os.path.join(DATA_DIR, 'simplex.pkl')

    parser = argparse.ArgumentParser(description=DOCSTR)
    parser.add_argument('-w', '--n_words_per_doc', type=int, default=5)
    parser.add_argument('-d', '--n_docs', type=int, default=2)
    parser.add_argument('-a', '--alpha', type=float, default=.1)
    parser.add_argument('-b', '--beta', type=float, default=.1)
    parser.add_argument('-f', '--filename', type=str, default=filename)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--n_steps', type=int, default=5,
                        help='Number intermediate distributions to calculate')
    parser.add_argument('--rotate', action='store_true', default=False,
                        help='Rotate topics in simplex rather than stretch')
    parser.add_argument('--plot_only', action='store_true', default=False,
                        help='Plot only; do not run experiment.')

    kwargs = vars(parser.parse_args())

    if not kwargs['plot_only']:
        del kwargs['plot_only']
        runexp(**kwargs)

    plotexp(kwargs['filename'])
