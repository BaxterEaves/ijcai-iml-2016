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


import numpy as np
import random
import itertools as it

from scipy.special import gammaln
from scipy.stats import dirichlet
from scipy.misc import logsumexp

from math import factorial
from math import log


def discrete_draw(p, n=1, logp=False):
    """
    Single draw from a discrete distribution defined by the probabilities p.

    Parameters
    ----------
    p : numpy.ndarray(n,)
        array of probabilities or log probabilities
    n : int, optional
        number od draws to do
    logp : bool
        True if p is a vector of log probabilities

    Returns
    -------
    idxs : int if n==1 otherwise, list<int>
        thhe indices in [0,...,len(p)-1] drawn from p
    """
    if logp:
        u = np.exp(p-logsumexp(p))
    else:
        u = p/np.sum(p)

    rs = np.random.rand(n)
    idxs = np.digitize(rs, np.cumsum(u/np.sum(u)))
    if n == 1:
        return idxs[0]
    else:
        return idxs


def loglinspace(a, b, n):
    """ Generate linearly separated integers in log space.

    Parameters
    ----------
    a : int
        start point
    b : int
        end point
    n : int
        number of steps

    Returns
    -------
    x : numpy.ndarray(1, <=n)
        A list of entries. If n is too large, x will have fewer than n entries.
    """
    x = np.exp(np.linspace(log(a), log(b), n))
    x = np.unique(np.array(np.round(x), dtype=int))
    if len(x) < n:
        print('Warning: loglinspace dropped entries')
    return x


def log_dirdisc_pdf(counts, alpha):
    """
    Log probability of a counts vector under Dirichlet-discrete.

    Paramters
    ---------
    counts : numpy.ndarray (n,)
        a n_categories-length vector of counts. Each bin represents the number
        of data points assigned to that bin.
    alpha : float (0, Inf)
        Symmetric Dirichlet distribution parameter.

    Returns
    -------
    logp : float
        log probability log(p(counts|alpha))
    """
    k = float(len(counts))
    n = float(np.sum(counts))
    a = k*alpha
    lg = np.sum(gammaln(counts+alpha))

    logp = gammaln(a) - gammaln(a + n) + lg - k*gammaln(alpha)
    return logp


def log_discrete_pdf(counts, phi):
    if counts.shape != phi.shape:
        raise ValueError('counts and phi must be the same size')
    return np.sum(counts*np.log(phi))


def gen_docs(n_docs=None, n_topics=None, n_words=None, n_words_per_doc=10,
             alpha=.1, beta=.1, noise=.0, phis=None, thetas=None):
    """ Generate documents from LDA generative process.

    Parameters
    ----------
    n_docs : int, optional if both phis and thetas are not None
        Number of documents to generate
    n_topics : int, optional if both phis and thetas are not None
        Number of topics
    n_words : int, optional if both phis and thetas are not None
        Number of words in the vocabulary
    n_words_per_doc : int, or array-like(int)
        Number of words to generate for each document
    alpha : float (0, Inf)
        Symmetric dirichlet parameter for the distribution of topics in
        documents.
    beta : float (0, Inf)
        Symmteric dirichlet parameter for the distribution of words in topics.
    noise : float [0, 1], optional
        Proportion of generated words to scramble.

    Returns
    -------
    docs : list
        List of data structures (dict) represtenting the generated documents.
        Each doc has a 'counts' key where each key is an integer word index and
        each value is the number of times that word occurs in the document.
    phis : list
        List of topics (weights of words)
    """
    if phis is None:
        phis = []
        for j in range(n_topics):
            phis.append(dirichlet.rvs([beta]*n_words)[0])
    else:
        n_words = len(phis[0])
        if not all(len(phi) == n_words for phi in phis):
            raise ValueError("All arrays in phis must be the same length")

        n_topics = len(thetas[0])
        if not all(len(theta) == n_topics for theta in thetas):
            raise ValueError("All arrays in thetas must be the same length")

        if len(phis) != n_topics:
            # FIXME: This is stupid error message
            raise ValueError("phis and thetas are inconsistent")

        n_docs = len(thetas)

    if isinstance(n_words_per_doc, int):
        n_words_per_doc = [n_words_per_doc]*n_docs

    if isinstance(n_words_per_doc, (list, np.ndarray)):
        if len(n_words_per_doc) != n_docs:
            raise ValueError("n_docs and n_words_per_doc are inconsistent")

    docs = []
    for k in range(n_docs):
        if thetas is None:
            theta_k = dirichlet.rvs([alpha]*n_topics)[0]
        else:
            theta_k = thetas[k]
        doc = {
            'words': set([i for i in range(n_words)]),
            'counts': dict((i, 0) for i in range(n_words)),
            'w': [],
            'theta': theta_k
        }
        for _ in range(n_words_per_doc[k]):
            j = int(discrete_draw(theta_k))
            w = int(discrete_draw(phis[j]))
            if noise > 0:
                if np.random.rand() < noise:
                    w = random.randrange(n_words)
            doc['counts'][w] += 1
            doc['w'].append(w)
        docs.append(doc)

    return docs, phis


def cosdist(a, b):
    """ 1 minus the cosine of the vectors a and b.

    If a and b are equal, returns 0. The maixumum distance is 2.
    """
    return 1 - np.dot(a, b)/(np.sum(a*a)**.5 * np.sum(b*b)**.5)


def docset_distance(docs_lst_a, docs_lst_b, n_words, row_agnostic=False):
    """ The distance cosine between two set of documents.

    Used to find a document set, among a fixed number of sets, that is
    similar to a hypothetical document set generated by the teaching model.

    Parameters
    ----------
    docs_lst_a : list of list(int)
        The first document set. `docs_lst_a`[d][i] is the word id of the ith
        word in document d.
    docs_lst_b : list of list(int)
        The second document set. `docs_lst_b`[d][i] is the word id of the ith
        word in document d.
    n_words : int
        The total number of words in the vocabulary (lexicon).
    row_agnostic : bool, optional
        If True, the documents in `docs_lst_a` and `docs_list_b` share an
        unknown correspondence. For example docs_list_a[d] may correspond to
        any document in docs_list_b. In this case, the documents are permuted
        and the minimum distance is returned.

    Returns
    -------
    dist : float
        The distance between the two document sets.
    """
    if len(docs_lst_a) != len(docs_lst_b):
        raise ValueError("docs_lst_a and docs_lst_b should be the same length"
                         " (contain the same number of documents).")

    def docs2counts(docs_lst):
        n_docs = len(docs_lst)
        counts = np.zeros((n_docs, n_words,))
        for didx, doc in enumerate(docs_lst):
            for word in doc:
                counts[didx, word] += 1.0
        # make the entire vector sum to one so that the documents are weighted
        # by their length.
        return counts

    ct_ary_a = docs2counts(docs_lst_a).flatten()
    if row_agnostic:
        ct_lst_b = docs2counts(docs_lst_b).tolist()
        dists = np.zeros(factorial(len(ct_lst_b)))
        for i, ct_lst_b_perm in enumerate(it.permutations(ct_lst_b)):
            dists[i] = cosdist(ct_ary_a, np.array(ct_lst_b_perm).flatten())
        dist = min(dists)
    else:
        ct_ary_b = docs2counts(docs_lst_b).flatten()
        dist = cosdist(ct_ary_a, ct_ary_b)

    return dist


def isess(weights):
    """ Effective sample size for importance sampler. """
    w = np.array(weights)
    n = float(len(weights))
    return n/(1.0 + np.var(w))


def isrelerr(weights):
    """ Relative error of importance sampling weights """
    n = float(len(weights))

    if n < 2:
        raise ValueError('Must have at least 2 weights to calculate error')

    w = np.array(weights)
    top = logsumexp(2.0*w) - np.log(n)
    bottom = 2.0*logsumexp(w) - 2.0*np.log(n)

    relerr = (1.0/n * (np.exp(top-bottom) - 1.0))**.5

    return relerr


def lda_topic_err(topics, n_kw, alpha):
    """ Error in the LDA topic predictive distribution.

    Parameters
    ----------
    topics : array-like(float)
        A n_topics by n_words (in vocabulary) where topics[k][w] is the
        probability of word w in topic k.
    n_kw : numpy.ndarray
        A member of the LDA class, n_kw[k][w] is the number of times word w
        is assigned to topic k.
    alpha : float (0, Inf)
        The symmetric Dirichlet prior on topics.

    Implementation notes
    --------------------
    Because the topic labels in `topics` and `n_kw` are not guaranteed to
    align, we calculate the error over all row permutations of n_kw and return
    the minimum.

    Returns
    -------
    error : float
        Sum of squarred error.
    """

    n_topics = len(topics)
    topics = np.array(topics)

    if len(n_kw) != n_topics:
        raise ValueError("n_kw and topics should be the same size.")

    topics_inferred = []
    for nkw in n_kw:
        topics_inferred.append(((nkw + alpha)/np.sum(nkw+alpha)).tolist())

    topics = np.array(topics)
    errs = np.zeros(factorial(n_topics))
    for i, tpcsinf in enumerate(it.permutations(topics_inferred)):
        # errtmp = (topics-np.array(tpcsinf))**2.0  # Euclidian distance
        errtmp = [cosdist(topics[i, :], np.array(tpcsinf[i]))
                  for i in range(n_topics)]
        errs[i] = np.sum(errtmp)

    return min(errs)


def euc2bary(p, q):
    """ Compute the barycentric coordinates of point p in the polygon q.

    Implementation Notes
    --------------------
    Meyer, M., Barr, A., Lee, H., & Desbrun, M. (2002). Generalized barycentric
    coordinates on irregular polygons. Journal of graphics tools, 7(1), 13-22.

    Parameters
    ----------
    p : numpy.ndarray
        2-dimensional coordinate of p in the polygon q
    q : array-like
        list or array of 2-dimenonal point forming a convex polygon.

    Returns
    -------
    w : numpy.ndarrary
        q-length array of barycentric coordinates
    """
    def _cotangent(a, b, c):
        ba = a - b
        bc = c - b
        return np.dot(bc, ba)/np.abs(np.cross(bc, ba))

    n = len(q)
    w = np.zeros(n)

    # check if p is a vertex of q
    qp = np.sum(np.abs(q-p), axis=1)
    vert = np.nonzero(qp == 0)[0]

    if vert.size == 1:
        pass
        w[vert[0]] = 1.
        return w
    elif vert.size > 1:
        raise IndexError('Found too many matching verticies.')

    for j, qj in enumerate(q):
        prv = (j + n - 1) % n
        nxt = (j + 1) % n
        w[j] = _cotangent(p, qj, q[prv]) + _cotangent(p, qj, q[nxt])
        w[j] /= np.sum((p-qj)**2.)

    assert not any(w < 0)

    w /= np.sum(w)
    return w


def bary2euc(w, q):
    """ Convert barycentric coordinates on the polygon q to Euclidean.

    Parameters
    ----------
    w : numpy.ndarray
        The  barycentric coordinates of of a point in the poylgon q.
    q : array-like
        list or array of 2-dimenonal point forming a convex polygon.

    Returns
    -------
    p : numpy.ndarrary
        2-dimensional point inside the polygon q.
    """
    p = np.sum(w[:, np.newaxis] * q, axis=0)
    return p
