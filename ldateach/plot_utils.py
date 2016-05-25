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

import math
import numpy as np
import seaborn as sns

from matplotlib import cbook
from matplotlib.colors import Normalize
from matplotlib import pyplot as plt
from matplotlib import path as mplPath
from mpl_toolkits.mplot3d import Axes3D

from ldateach.utils import euc2bary
from ldateach.utils import bary2euc
from ldateach.fastlda import teach_lda_pgis as pgis

from numpy import ma
from scipy.misc import logsumexp
from multiprocessing.pool import Pool


def pp_plot(f, p, nbins, ax=None):
    """ P-P plot of the empirical CDFs of values in two lists, f and p. """
    if ax is None:
        ax = plt.gca()

    uniqe_vals_f = list(set(f))
    uniqe_vals_p = list(set(p))

    combine = uniqe_vals_f
    combine.extend(uniqe_vals_p)
    combine = list(set(combine))

    if len(uniqe_vals_f) > nbins:
        bins = nbins
    else:
        bins = sorted(combine)
        bins.append(bins[-1]+bins[-1]-bins[-2])

    ff, edges = np.histogram(f, bins=bins, density=True)
    fp, _ = np.histogram(p, bins=edges, density=True)

    Ff = np.cumsum(ff*(edges[1:]-edges[:-1]))
    Fp = np.cumsum(fp*(edges[1:]-edges[:-1]))

    plt.plot([0, 1], [0, 1], c='dodgerblue', lw=2, alpha=.8)
    plt.plot(Ff, Fp, c='black', lw=2, alpha=.9)
    plt.xlim([0, 1])
    plt.ylim([0, 1])


def stacked_bar(ax, x, heights, label=None):
    """ Plots a stacked barchart.

    Parameters
    ----------
    ax : matpliotlib.Axis
        The axis on which to draw
    x : numpy.ndarray or list
        The left positions of each bar
    heights : numpy.ndarrary or list
        The heights of the bars
    label : str, optional
        Label for the series. Each set of bars will be labeled label_i. For
        example, if label='Doc' bars will be labeled 'Doc 0', 'Doc 1', ...
    """
    colors = sns.color_palette('muted', len(x))
    bottom = np.zeros(len(x))
    label_i = None
    for i, y in enumerate(heights):
        if label:
            label_i = label + ' ' + str(i)
        ax.bar(x, y, bottom=bottom, label=label_i, facecolor=colors[i])
        bottom += y
    return ax


class MidpointNorm(Normalize):
    def __init__(self, midpoint=0, vmin=None, vmax=None, clip=False):
        Normalize.__init__(self, vmin, vmax, clip)
        self.midpoint = midpoint

    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip

        result, is_scalar = self.process_value(value)

        self.autoscale_None(result)
        vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint

        if not (vmin < midpoint < vmax):
            raise ValueError("midpoint must be between maxvalue and minvalue.")
        elif vmin == vmax:
            result.fill(0)  # Or should it be all masked? Or 0.5?
        elif vmin > vmax:
            raise ValueError("maxvalue must be bigger than minvalue")
        else:
            vmin = float(vmin)
            vmax = float(vmax)
            if clip:
                mask = ma.getmask(result)
                result = ma.array(np.clip(result.filled(vmax), vmin, vmax),
                                  mask=mask)

            # ma division is very slow; we can take a shortcut
            resdat = result.data

            # First scale to -1 to 1 range, than to from 0 to 1.
            resdat -= midpoint
            resdat[resdat > 0] /= abs(vmax - midpoint)
            resdat[resdat < 0] /= abs(vmin - midpoint)

            resdat /= 2.
            resdat += 0.5
            result = ma.array(resdat, mask=result.mask, copy=False)

        if is_scalar:
            result = result[0]
        return result

    def inverse(self, value):
        if not self.scaled():
            raise ValueError("Not invertible until scaled")
        vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint

        if cbook.iterable(value):
            val = ma.asarray(value)
            val = 2 * (val-0.5)
            val[val > 0] *= abs(vmax - midpoint)
            val[val < 0] *= abs(vmin - midpoint)
            val += midpoint
            return val
        else:
            val = 2 * (val - 0.5)
            if val < 0:
                return val*abs(vmin-midpoint) + midpoint
            else:
                return val*abs(vmax-midpoint) + midpoint


# ---
def _concentric_poly_fill(n, steps):
    """ Unit n-gon centered at the origin. Interior points are placed in
    concentric n-gons. The total number of points is n*steps.
    """
    if n < 3:
        raise ValueError('n must be >= 3')

    segrads = math.pi*2./n
    radstep = 1./steps

    x = []
    y = []
    for step, rad in enumerate(np.linspace(radstep, 1, steps)[::-1]):
        rot = segrads*step/2.
        x.extend([math.sin(segrads * i + rot) * rad for i in range(n)])
        y.extend([math.cos(segrads * i + rot) * rad for i in range(n)])

    x = np.array(x + [0])
    y = np.array(y + [0])

    x_outer = x[:n]
    y_outer = y[:n]

    p = np.array([[xi, yi] for xi, yi in zip(x_outer, y_outer)])

    return p, x, y


def _uniform_poly_fill(n, g):
    """ Unit n-gon centered at the origin. Interior points are placed on a
    g-by-g grid. Points outside the polygon are rejected, thus the total number
    of points is less than g^2.
    """

    if n < 3:
        raise ValueError('n must be >= 3')

    segrads = math.pi*2./n
    r = np.array([segrads * i for i in range(n)])
    x = np.sin(r)
    y = np.cos(r)

    path = mplPath.Path(np.array([[xi, yi] for xi, yi in zip(x, y)]))

    pts = np.linspace(-1, 1, g)

    xe = []
    ye = []
    for x2 in pts:
        for y2 in pts:
            if path.contains_point((x2, y2)):
                xe.append(x2)
                ye.append(y2)

    p = np.array([[xi, yi] for xi, yi in zip(x, y)])

    return p, np.append(x, xe), np.append(y, ye)


def _compute_logp_inr(q, p, n_words, topics, alpha=.1, beta=.1, n_samples=5000,
                      seed=None):
    doc_cts = np.array(np.round(n_words*euc2bary(p, q)), dtype=int)
    w = []
    for wrd, ct in enumerate(doc_cts):
        w.extend([wrd]*ct)
    docs = [{'w': w}]
    logp = pgis(docs, topics, alpha=alpha, beta=beta, n_samples=n_samples,
                seed=seed)
    return logp


def _compute_logp_otr(args):
    return _compute_logp_inr(*args)


def ldateachviz(tpcs, fill_type='concentric', **kwargs):
    """ Visualize teaching probability distribution for topic models of
    arbitrary vocabulary size.

    Parameters
    ----------
    tpcs : list(numpy.ndarray)
        list of topics int the topic model
    fill_type : str
        method of fill for the polygon. 'concentric' fills the polygon with
        `step` concentric rings composed of n_words point; uniform fills the
        polygon with points from `g` by `g` grid.
    n_words_doc : int
        number of words in the evluated documents. 50 by default.
    n_samples : int
        number of samples used for the teaching probability approximation.
        1000 by default.
    alpha : float (0, Inf)
        LDA alpha parameter. Default is 0.1.
    beta : float (0, Inf)
        LDA beta parameter. Default is 0.1.

    Other parameters
    ----------------
    steps : int
        If `fill_type` is concentric, the number of concentric polygons to use.
    g : int
        If 'fill_type` is uniform, the polygon will be filled with points from
        a `g` by `g` grid.
    """
    tpcs_sum = np.sum(tpcs, axis=0)
    tpcs_sum /= np.max(tpcs_sum)
    idxs = np.argsort(tpcs_sum)[::-1]
    topics = tpcs[:, idxs]
    tpcs_sum = tpcs_sum[idxs]

    n_words = len(topics[0])
    n_words_doc = kwargs.get('n_words_doc', 50)
    n_samples = kwargs.get('n_samples', 1000)
    alpha = kwargs.get('alpha', .1)
    beta = kwargs.get('beta', .1)

    if not kwargs.get('use_mp', False):
        mapper = lambda func, args: [func(arg) for arg in args]
    else:
        pool = Pool()
        mapper = pool.map

    if fill_type == 'concentric':
        steps = kwargs.get('steps', n_words-1)
        q, x, y = _concentric_poly_fill(n_words, steps)
    elif fill_type == 'uniform':
        g = kwargs.get('g', n_words)
        q, x, y = _uniform_poly_fill(n_words, g)
    else:
        raise ValueError("fill_type must be 'concentric' or 'uniform'.")

    pts = np.array([[xi, yi] for xi, yi in zip(x, y)])
    topics_xy = np.array([bary2euc(w, q) for w in topics])

    args = [(q, p, n_words_doc, topics, alpha, beta, n_samples) for p in pts]
    res = mapper(_compute_logp_otr, args)
    res = res-logsumexp(res)
    max = np.argmax(res)
    pmax = [x[max], y[max]]

    fig = kwargs.get('fig', None)
    if fig is None:
        fig = plt.figure(tight_layout=True, figsize=(6.5, 3.5), dpi=300)

    ax = fig.add_subplot(1, 2, 1)
    for wrd_height, (qx, qy) in zip(tpcs_sum, q):
        tick_len = 1+.25*wrd_height
        plt.plot([0, qx*tick_len], [0, qy*tick_len], lw=.5, zorder=0,
                 color='#bbbbbb')

    plt.tricontourf(x, y, res, 200, cmap='viridis', zorder=1)
    plt.scatter(topics_xy[:, 0], topics_xy[:, 1], color='#333333',
                marker='+', linewidth=1, s=64, zorder=2)
    plt.scatter([pmax[0]], [pmax[1]], color='#333333', s=36, zorder=3)
    plt.axis('off')
    plt.xlim([-1.25, 1.25])
    plt.ylim([-1.25, 1.25])

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.plot_trisurf(x, y, res, cmap=plt.cm.viridis)
