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


DOCSTR = """ IMDB experiment.
Calculates the teaching probability, likleihood, and cosine distance from the
target model for each synopsis in the IMDB top 250.
"""

import pickle
import sys
import os
import random
import pandas
import itertools as it
import numpy as np

from scipy.misc import logsumexp
from scipy.stats import pearsonr
from multiprocessing.pool import Pool
from ldateach.fastlda import teach_lda_pgis
from ldateach.fastlda import FastLDA
from ldateach.utils import cosdist

import matplotlib as mpl
mpl.use('Agg')

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context('paper')

DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(DIR, 'data')
SCRAPE_DIR = os.path.join(DIR, '..', 'scrape', 'data')


def align_yaxis(ax1, v1, ax2, v2):
    """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny+dy, maxy+dy)


def _calc_docset_logp_inner(docs, topics, topics_list, α, β, n_samples, seed):
    pt = teach_lda_pgis(docs, topics, topics_list=topics_list, alpha=α, beta=β,
                        n_samples=n_samples, seed=seed, return_parts=True)
    prior, numer, denom = pt

    lp = prior + logsumexp(numer) - logsumexp(denom)
    ll = logsumexp(numer) - np.log(len(numer))

    print(".", end="")
    sys.stdout.flush()
    return lp, ll


def _calc_docset_logp(args):
    argsin = args[:-1]
    didx = args[-1]
    return _calc_docset_logp_inner(*argsin), didx


def runexp(imdb_filename=None, res_filename='res.pkl', alpha=.1, beta=.1,
           n_teach_docs=1, topics_list=None, n_topics=None, n_samples=1000,
           n_runs=32, seed=1337, **kwargs):
    kwargsin = locals()

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    metadata = pickle.load(open(imdb_filename, 'rb'))
    docs = metadata['docs']
    genres = metadata['genres']
    words = metadata['words']

    if n_topics is None:
        n_topics = len(genres)
    n_words = len(words)
    n_docs = len(docs)

    # generate topics from the database
    lda = FastLDA(docs, n_topics, n_words, alpha, beta, seed=seed)
    lda.run(1000, optimize=False)
    topics = lda.topic_word_distributions()

    cdist = []
    if topics_list is None:
        normed_topics = np.sum(topics, axis=0)
    else:
        normed_topics = np.array(topics)[tuple(topics_list), :]
        normed_topics = np.sum(normed_topics, axis=0)
    normed_topics /= np.sum(normed_topics)

    pool = Pool()
    didxs = []
    logps = []
    logls = []
    word_count = []
    for run in range(n_runs):
        args = []
        for didx in it.combinations(range(n_docs), n_teach_docs):
            seed = random.randrange(2**31)
            args.append(([docs[i] for i in didx], topics, topics_list, alpha,
                         beta, n_samples, seed, didx,))

        print("* Caclulating docset probabilities (%d/%d)" % (run+1, n_runs,))
        out = pool.map(_calc_docset_logp, args)
        print("Run %d done." % (run+1,))
        for (lp, ll), didx in out:
            doc = docs[didx[0]]
            wbinned = np.array(np.bincount(doc['w'], minlength=n_words),
                               dtype=float)
            wbinned /= float(np.sum(wbinned))
            dist = cosdist(wbinned, normed_topics)

            assert not (np.isnan(dist) or np.isinf(dist))

            word_count.append(len(doc['w']))
            cdist.append(dist)
            logps.append(lp)
            logls.append(ll)
            didxs.append(didx)

    assert len(logps) == n_runs*len(docs)
    assert len(logls) == n_runs*len(docs)
    assert len(cdist) == n_runs*len(docs)
    assert len(didxs) == n_runs*len(docs)

    result = {
        'args': kwargsin,
        'didxs': didxs,
        'logps': logps,
        'logls': logls,
        'cosdist': cdist,
        'topics': topics,
        'word_count': word_count,
        'doc_topic_dist': lda.doc_topic_distributions(),
        'topic_word_dist': lda.topic_word_distributions(),
        'n_dk': lda.n_dk,
        'n_kw': lda.n_kw,
        'n_k': lda.n_k,
    }

    pickle.dump(result, open(res_filename, 'wb'))


def plotexp(res_filename, figname, figtitle='film'):
    result = pickle.load(open(res_filename, 'rb'))
    metadata = pickle.load(open(result['args']['imdb_filename'], 'rb'))

    top_n_words = 50
    n_words = len(metadata['words'])

    topic_order = np.argsort(result['n_k'])[::-1]
    n_topics = len(topic_order)
    words = [metadata['idx_to_word'][widx] for widx in range(n_words)]

    # Plot each topic
    fig = plt.figure()
    base_figname = os.path.join(DATA_DIR, 'topics_%d' % n_topics)
    for i, tidx in enumerate(topic_order):
        topic_prop = result['n_k'][tidx]/sum(result['n_k'])
        topic_dist = result['topic_word_dist'][tidx]
        topic_data = [{'word': w, 'p': p} for w, p in zip(words, topic_dist)]
        df = pandas.DataFrame(topic_data)
        df = df.sort('p', ascending=False).head(top_n_words)
        sns.barplot(data=df, x='word', y='p', color="#333333")
        plt.ylim([min(df['p'])*.95, max(df['p'])*1.05])
        plt.xticks(rotation=90)
        plt.title('Topic %d, p: %f' % (tidx, topic_prop,))
        plt.savefig(os.path.join(base_figname, 't%d.png' % (tidx,)), dpi=300)
        fig.clf()

    tpc_data = []
    titles = []
    for didx in range(len(result['n_dk'])):
        titles.append(metadata['docs'][didx]['film'])
        datum = {}
        sm = np.sum(result['n_dk'][didx])
        for tidx, ct in enumerate(result['n_dk'][didx]):
            datum['t_%d' % tidx] = ct/sm
        tpc_data.append(datum)

    kd_data = pandas.DataFrame(tpc_data, index=titles)

    data = []
    for idx, lp, ll, cd, wc in zip(result['didxs'], result['logps'],
                                   result['logls'], result['cosdist'],
                                   result['word_count']):
        datum = {
            'title': ', '.join([metadata['docs'][i]['film'] for i in idx]),
            'logp': lp,
            'logl': ll,
            'n_words': wc,
            'cdist': cd,
        }
        data.append(datum)
    df = pandas.DataFrame(data).sort('logp', ascending=False)

    data = []
    for title in df['title'].unique():
        mlp = np.mean(df['logp'][df['title'] == title])
        mll = np.mean(df['logl'][df['title'] == title])
        mcd = np.mean(df['cdist'][df['title'] == title])
        mwc = np.mean(df['n_words'][df['title'] == title])
        elp = np.std(df['logp'][df['title'] == title])/result['args']['n_runs']**.5
        ell = np.std(df['logl'][df['title'] == title])/result['args']['n_runs']**.5
        dat = {
            'title': title,
            'logP Mean': mlp,
            'logP Error': elp,
            'logL Mean': mll,
            'logL Error': ell,
            'cosine distance': mcd,
            'words': mwc,
        }
        data.append(dat)

    dfo = pandas.DataFrame(data)
    dfo['logP Mean'] -= logsumexp(dfo['logP Mean'])
    dfo['logL Mean'] -= logsumexp(dfo['logL Mean'])

    dfo.sort('logP Mean', ascending=False, inplace=True)

    plt.figure(figsize=(4.5, 4.5))
    tpc_ticks = ['%d' % (idx,) for idx in topic_order]
    kd_data = kd_data[['t_%d' % (idx,) for idx in topic_order]]

    ticks = []
    maxlen = 25
    head = 20
    for t in dfo['title']:
        if len(t) > maxlen:
            tick = t[:maxlen].strip() + '…'
        else:
            tick = t
        ticks.append(tick)

    ax = plt.subplot(2, 1, 1)
    vmax = 0.45  # np.max(kd_data.values)
    norm = mpl.colors.PowerNorm(.65, 0, vmax)
    sns.heatmap(kd_data.loc[dfo['title']].head(head), cmap="CMRmap",
                cbar_kws={'shrink': .8}, vmin=0, vmax=vmax, ax=ax, norm=norm)
    filmticks = ticks[:head]
    ax.set_xticklabels([])
    ax.set_ylabel('Best')
    ax.set_title(figtitle)
    ax.set_yticklabels(filmticks[::-1], fontsize=6)

    ax = plt.subplot(2, 1, 2)
    sns.heatmap(kd_data.loc[dfo['title']].tail(head), cmap="CMRmap",
                cbar_kws={'shrink': 0}, vmin=0, vmax=vmax, ax=ax, norm=norm)
    filmticks = ticks[-head:]
    ax.set_xticklabels(tpc_ticks, rotation=90, fontsize=6)
    ax.set_yticklabels(filmticks[::-1], fontsize=6)
    ax.set_xlabel('Topic')
    ax.set_ylabel('Worst')
    dtfigname = os.path.join(DATA_DIR, 'topics_%d' % n_topics, 'dtop.png')
    plt.savefig(dtfigname, dpi=300)

    colors = ("#34495e", "#95a5a6", "#e74c3c", "#2ecc71",)
    x = np.arange(len(dfo))
    plt.figure(figsize=(8.5, 3.5), facecolor='white')
    host = plt.gca()
    plt.subplots_adjust(right=0.75)

    axes = [host, host.twinx(), host.twinx()]
    axes[2].tick_params(axis='y', direction='out', pad=40)

    axes[2].errorbar(y=dfo['cosine distance'].values, x=x,
                     fmt='o', label='Cosine distance', c=colors[2],
                     mec=colors[2], mfc=colors[2], ecolor=colors[2], ms=3)
    axes[1].errorbar(y=dfo['logL Mean'].values, x=x, yerr=dfo['logL Error'],
                     fmt='o', label='Likelihood', c=colors[1], mec=colors[1],
                     mfc=colors[1], ecolor=colors[1], ms=3)
    axes[0].errorbar(y=dfo['logP Mean'].values, x=x, yerr=dfo['logP Error'],
                     fmt='o', label='Teaching', c=colors[0], mfc=colors[0],
                     mec=colors[0], ecolor=colors[0], ms=3)

    # axes[3].set_ylabel('Number of words', color=colors[3])
    axes[2].set_ylabel('Cosine Distance', color=colors[2])
    axes[1].set_ylabel('Likelihood', color=colors[1])

    axes[0].set_ylabel('%s\nMean teaching probability' % (figtitle,),
                       color=colors[0])
    axes[0].tick_params(axis='y', colors=colors[0])

    for i, ax in enumerate(axes[1:]):
        ax.tick_params(axis='y', colors=colors[i+1])
        ax.grid(color='none')
        ax.set_xticks(x)
        ax.set_xticklabels(['' for _ in range(len(x))])

    top_n = 20
    print('TOP %d TEACHING MOVIES' % (top_n,))
    for i in range(top_n):
        print('\t{}. {}'.format(i+1, ticks[i]))

    # make sure all ytick have 0 at the same spot and the same # of ticks
    # XXX: if the teaching distribution is very flat compared to the
    # likelihood, this may cause it not to show up.
    topics_list = result['args']['topics_list']
    if topics_list is not None:
        if len(topics_list) > .5*result['args']['n_topics']:
            align_yaxis(axes[1], 0.0, axes[0], 0.0)

    # make sure that the tick labels don't overlap
    max_ticks = 100  # only display some movie title
    n_x_ticks = len(ticks)
    tick_step = int(n_x_ticks/max_ticks)
    xtick_labels = ticks[:-1:tick_step]
    xticks = [i for i in range(0, n_x_ticks, tick_step)]
    assert len(xticks) == len(xtick_labels)
    axes[0].set_xticks(xticks)
    axes[0].set_xticklabels(xtick_labels, rotation=90, fontsize=5)

    plt.xlabel('Film')
    plt.savefig(figname, dpi=300)

    # scatter plots are correlations for
    fig, axes = plt.subplots(1, 5)
    fig.set_size_inches(15.5, 3.5)
    df['Likelihood'] = np.exp(df['logl'])

    sns.regplot(x='logl', y='logp', data=df, ax=axes[0])
    axes[0].set_xlabel('Log likelihood')
    axes[0].set_ylabel('Log teaching probability')

    corres = pearsonr(df['logl'].values, df['logp'])
    print('PEARSON Likelihood v Teaching probability')
    print('  r={}, p={}'.format(*corres))

    sns.regplot(x='logl', y='cdist', data=df, ax=axes[1])
    axes[1].set_xlabel('Log likelihood')
    axes[1].set_ylabel('Cosine distance')

    corres = pearsonr(df['logl'].values, df['cdist'])
    print('PEARSON Likelihood v Cosine dist')
    print('  r={}, p={}'.format(*corres))

    # set axis limits to min/max -/+ standard deviation
    def _stdaxlim(dtfrm, colx, coly, ax):
        ax.set_ylim([dtfrm[coly].min()-dtfrm[coly].std(),
                     dtfrm[coly].max()+dtfrm[coly].std()])
        ax.set_xlim([dtfrm[colx].min()-dtfrm[colx].std(),
                     dtfrm[colx].max()+dtfrm[colx].std()])

    sns.regplot(x='logp', y='cdist', data=df, ax=axes[2])
    _stdaxlim(df, 'logp', 'cdist', axes[2])
    axes[2].set_xlabel('Log teaching probability')
    axes[2].set_ylabel('Cosine distance')

    corres = pearsonr(df['logp'].values, df['cdist'])
    print('PEARSON Teaching probabiliyt  v Cosine dist')
    print('  r={}, p={}'.format(*corres))

    sns.regplot(x='logl', y='n_words', data=df, ax=axes[3])
    _stdaxlim(df, 'logl', 'n_words', axes[3])
    axes[3].set_xlabel('Log  likelihood')
    axes[3].set_ylabel('Number of words')
    plt.savefig(os.path.join(DATA_DIR, 'imdb_corr.png'), dpi=300)

    corres = pearsonr(df['logl'].values, df['n_words'])
    print('PEARSON Likelihood v # Words')
    print('  r={}, p={}'.format(*corres))

    sns.regplot(x='logp', y='n_words', data=df, ax=axes[4])
    _stdaxlim(df, 'logp', 'n_words', axes[4])
    axes[4].set_xlabel('Teaching probability')
    axes[4].set_ylabel('Number of words')

    corres = pearsonr(df['logp'].values, df['n_words'])
    print('PEARSON Teaching probability v # Words')
    print('  r={}, p={}'.format(*corres))


if __name__ == "__main__":
    import argparse

    imdb_filename = os.path.join(SCRAPE_DIR, 'imdb', 'imdb1k.csv.pkl')
    res_filename = os.path.join(DATA_DIR, 'exp-imdb-top-1k.pkl')

    # Default number of topics derived from exp_model_selection.py.
    parser = argparse.ArgumentParser(description=DOCSTR)
    parser.add_argument('-b', '--beta', type=float, default=.1)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('-l', '--topic_label', type=str, default='All')
    parser.add_argument('--imdb_filename', type=str, default=imdb_filename,
                        help="path to imdb data file")
    parser.add_argument('--res_filename', type=str, default=res_filename,
                        help="save file name")
    parser.add_argument('-t', '--n_topics', type=int, default=37,
                        help='Number of topics in the target model')
    parser.add_argument('-r', '--n_runs', type=int, default=8,
                        help='Number of runs to average over')
    parser.add_argument('--n_samples', type=int, default=10000,
                        help='Number of samples for approximation.')
    parser.add_argument('--topics_list', type=int, nargs='+', default=None,
                        help='List of topics to teach. Defaults to all topics.')
    parser.add_argument('--plot_only', action='store_true', default=False,
                        help='Plot only; do not run experiment.')

    kwargs = vars(parser.parse_args())

    # This value of α reflects the Procedure used in 'Finding Scientific
    # Topics'. This value of α is also used in exp_model_selection.py to choose
    # the number of topics.
    kwargs['alpha'] = 50.0/kwargs['n_topics']

    if not kwargs['plot_only']:
        del kwargs['plot_only']
        runexp(**kwargs)

    figname = kwargs['res_filename'] + '.png'
    plotexp(kwargs['res_filename'], figname, kwargs['topic_label'])
