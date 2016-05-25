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


from libcpp.vector cimport vector
from libcpp cimport bool as cbool
from scipy.misc import logsumexp
from multiprocessing.pool import Pool
from math import log

import itertools as it
import numpy
import random


LDATEACH_EXACT = 0
LDATEACH_UNIS = 1
LDATEACH_PGIS = 2


cdef extern from "lda.hpp":
    cdef cppclass LDA:
        LDA(vector[vector[size_t]] docs, size_t n_tpcs, size_t n_wrds,
            double alpha, double beta, int seed) except +

        void run(size_t n_steps, cbool optimize)

        vector[vector[size_t]] get_assignment()
        vector[size_t] get_asgn_radix()
        vector[size_t] get_n_k()
        vector[vector[size_t]] get_n_dk()
        vector[vector[size_t]] get_n_kw()
        double likelihood()

        vector[vector[double]] topic_word_distributions()

        vector[vector[double]] doc_topic_distributions()

        void set_asgn_from_radix(vector[size_t])

    cdef cppclass SparseLDA:
        SparseLDA(vector[vector[size_t]] docs, size_t n_tpcs, size_t n_wrds,
                  double alpha, double beta, int seed) except +

        void run(size_t n_steps, cbool optimize)

        vector[vector[size_t]] get_assignment()
        vector[size_t] get_asgn_radix()
        vector[size_t] get_n_k()
        vector[vector[size_t]] get_n_dk()
        vector[vector[size_t]] get_n_kw()
        double likelihood()

        vector[vector[double]] topic_word_distributions()

        vector[vector[double]] doc_topic_distributions()

        void set_asgn_from_radix(vector[size_t])

    cdef cppclass LazyLDA:
        LazyLDA(vector[vector[size_t]] docs, size_t n_tpcs, size_t n_wrds,
                double alpha, double beta, int seed) except +

        void run(size_t n_steps, cbool optimize)

        vector[vector[size_t]] get_assignment()
        vector[size_t] get_asgn_radix()
        vector[size_t] get_n_k()
        vector[vector[size_t]] get_n_dk()
        vector[vector[size_t]] get_n_kw()
        double likelihood()

        vector[vector[double]] topic_word_distributions()

        vector[vector[double]] doc_topic_distributions()

        void set_asgn_from_radix(vector[size_t])

    cdef cppclass LightLDA:
        LightLDA(vector[vector[size_t]] docs, size_t n_tpcs, size_t n_wrds,
                double alpha, double beta, int seed) except +

        void run(size_t n_steps, cbool optimize)

        vector[vector[size_t]] get_assignment()
        vector[size_t] get_asgn_radix()
        vector[size_t] get_n_k()
        vector[vector[size_t]] get_n_dk()
        vector[vector[size_t]] get_n_kw()
        double likelihood()

        vector[vector[double]] topic_word_distributions()

        vector[vector[double]] doc_topic_distributions()

        void set_asgn_from_radix(vector[size_t])

    cdef struct Estimator:
        double prior;
        vector[double] numer;
        vector[double] denom;

    cdef Estimator ldateach_exact(const vector[vector[size_t]]& docs,
                                  const vector[vector[double]]& tpcs,
                                  vector[size_t]& tpc_lst,
                                  double alpha, double beta)

    cdef Estimator ldateach_unis(const vector[vector[size_t]]& docs,
                                 const vector[vector[double]]& tpcs,
                                 vector[size_t]& tpc_lst,
                                 double alpha, double beta,
                                 size_t n_samples, unsigned int seed)

    cdef Estimator ldateach_pgis(const vector[vector[size_t]]& docs,
                                 const vector[vector[double]]& tpcs,
                                 vector[size_t]& tpc_lst,
                                 double alpha, double beta,
                                 size_t n_samples, unsigned int seed)

cdef class FastLDA:
    cdef LDA *lda_ptr;
    def __cinit__(self, docs, n_topics, n_words, alpha, beta, seed=-1):
        """
        Parameters
        ----------
        docs : list(dict)
            list of document data structures (see ldateach.utils.gen_docs)
        n_topics : int
            Number of topics
        n_words : int
            Number of words in the vocabulary
        alpha : float (0, Inf)
            Symmetric dirichlet parameter for the distribution of topics in
            documents.
        beta : float (0, Inf)
            Symmteric dirichlet parameter for the distribution of words in
            topics.
        seed : int (optional)
            Seed for the C++ RNG
        """
        docsw = [doc['w'] for doc in docs] 
        self.lda_ptr = new LDA(docsw, n_topics, n_words, alpha, beta, seed)

    def __dealloc__(self):
        del self.lda_ptr

    def run(self, n_steps=1, optimize=False):
        """ Run the sampler for n steps
        
        Parameters
        ----------
        n_steps : int
            Number of Gibbs sampling steps
        optimize : bool
            If True, use simmulated annealing on the Gibbs sampler to find a
            maxima. If run is called multiple times with optimize=True, the
            annealing schedule will be restarted each time.
        """
        self.lda_ptr.run(n_steps, optimize)

    def get_assignment(self):
        return self.lda_ptr.get_assignment()

    def set_assignment(self, asgn):
        radix = [item for sublist in asgn for item in sublist]
        self.lda_ptr.set_asgn_from_radix(radix)

    def get_radix(self):
        return self.lda_ptr.get_asgn_radix()

    def likelihood(self):
        return self.lda_ptr.likelihood()

    @property
    def n_dk(self):
        cts = self.lda_ptr.get_n_dk()
        return [numpy.array(ct) for ct in cts]

    @property
    def n_k(self):
        cts = self.lda_ptr.get_n_k()
        return cts

    @property
    def n_kw(self):
        cts = self.lda_ptr.get_n_kw()
        return [numpy.array(ct) for ct in cts]

    def topic_word_distributions(self):
        return self.lda_ptr.topic_word_distributions()

    def doc_topic_distributions(self):
        return self.lda_ptr.doc_topic_distributions()


cdef class FastLazyLDA:
    cdef LazyLDA *lda_ptr;
    def __cinit__(self, docs, n_topics, n_words, alpha, beta, seed=-1):
        """
        Parameters
        ----------
        docs : list(dict)
            list of document data structures (see ldateach.utils.gen_docs)
        n_topics : int
            Number of topics
        n_words : int
            Number of words in the vocabulary
        alpha : float (0, Inf)
            Symmetric dirichlet parameter for the distribution of topics in
            documents.
        beta : float (0, Inf)
            Symmteric dirichlet parameter for the distribution of words in
            topics.
        seed : int (optional)
            Seed for the C++ RNG
        """
        docsw = [doc['w'] for doc in docs] 
        self.lda_ptr = new LazyLDA(docsw, n_topics, n_words, alpha, beta, seed)

    def __dealloc__(self):
        del self.lda_ptr

    def run(self, n_steps=1, optimize=False):
        """ Run the sampler for n steps
        
        Parameters
        ----------
        n_steps : int
            Number of Gibbs sampling steps
        optimize : bool
            If True, use simmulated annealing on the Gibbs sampler to find a
            maxima. If run is called multiple times with optimize=True, the
            annealing schedule will be restarted each time.
        """
        self.lda_ptr.run(n_steps, optimize)

    def get_assignment(self):
        return self.lda_ptr.get_assignment()

    def set_assignment(self, asgn):
        radix = [item for sublist in asgn for item in sublist]
        self.lda_ptr.set_asgn_from_radix(radix)

    def get_radix(self):
        return self.lda_ptr.get_asgn_radix()

    def likelihood(self):
        return self.lda_ptr.likelihood()

    @property
    def n_dk(self):
        cts = self.lda_ptr.get_n_dk()
        return [numpy.array(ct) for ct in cts]

    @property
    def n_k(self):
        cts = self.lda_ptr.get_n_k()
        return cts

    @property
    def n_kw(self):
        cts = self.lda_ptr.get_n_kw()
        return [numpy.array(ct) for ct in cts]

    def topic_word_distributions(self):
        return self.lda_ptr.topic_word_distributions()

    def doc_topic_distributions(self):
        return self.lda_ptr.doc_topic_distributions()


cdef class FastLightLDA:
    cdef LightLDA *lda_ptr;
    def __cinit__(self, docs, n_topics, n_words, alpha, beta, seed=-1):
        """
        Parameters
        ----------
        docs : list(dict)
            list of document data structures (see ldateach.utils.gen_docs)
        n_topics : int
            Number of topics
        n_words : int
            Number of words in the vocabulary
        alpha : float (0, Inf)
            Symmetric dirichlet parameter for the distribution of topics in
            documents.
        beta : float (0, Inf)
            Symmteric dirichlet parameter for the distribution of words in
            topics.
        seed : int (optional)
            Seed for the C++ RNG
        """
        docsw = [doc['w'] for doc in docs] 
        self.lda_ptr = new LightLDA(docsw, n_topics, n_words, alpha, beta, seed)

    def __dealloc__(self):
        del self.lda_ptr

    def run(self, n_steps=1, optimize=False):
        """ Run the sampler for n steps
        
        Parameters
        ----------
        n_steps : int
            Number of Gibbs sampling steps
        optimize : bool
            If True, use simmulated annealing on the Gibbs sampler to find a
            maxima. If run is called multiple times with optimize=True, the
            annealing schedule will be restarted each time.
        """
        self.lda_ptr.run(n_steps, optimize)

    def get_assignment(self):
        return self.lda_ptr.get_assignment()

    def set_assignment(self, asgn):
        radix = [item for sublist in asgn for item in sublist]
        self.lda_ptr.set_asgn_from_radix(radix)

    def get_radix(self):
        return self.lda_ptr.get_asgn_radix()

    def likelihood(self):
        return self.lda_ptr.likelihood()

    @property
    def n_dk(self):
        cts = self.lda_ptr.get_n_dk()
        return [numpy.array(ct) for ct in cts]

    @property
    def n_k(self):
        cts = self.lda_ptr.get_n_k()
        return cts

    @property
    def n_kw(self):
        cts = self.lda_ptr.get_n_kw()
        return [numpy.array(ct) for ct in cts]

    def topic_word_distributions(self):
        return self.lda_ptr.topic_word_distributions()

    def doc_topic_distributions(self):
        return self.lda_ptr.doc_topic_distributions()


cdef class FastSparseLDA:
    cdef SparseLDA *lda_ptr;
    def __cinit__(self, docs, n_topics, n_words, alpha, beta, seed=-1):
        """
        Parameters
        ----------
        docs : list(dict)
            list of document data structures (see ldateach.utils.gen_docs)
        n_topics : int
            Number of topics
        n_words : int
            Number of words in the vocabulary
        alpha : float (0, Inf)
            Symmetric dirichlet parameter for the distribution of topics in
            documents.
        beta : float (0, Inf)
            Symmteric dirichlet parameter for the distribution of words in
            topics.
        seed : int (optional)
            Seed for the C++ RNG
        """
        docsw = [doc['w'] for doc in docs] 
        self.lda_ptr = new SparseLDA(docsw, n_topics, n_words, alpha, beta, seed)

    def __dealloc__(self):
        del self.lda_ptr

    def run(self, n_steps=1, optimize=False):
        """ Run the sampler for n steps
        
        Parameters
        ----------
        n_steps : int
            Number of Gibbs sampling steps
        optimize : bool
            If True, use simmulated annealing on the Gibbs sampler to find a
            maxima. If run is called multiple times with optimize=True, the
            annealing schedule will be restarted each time.
        """
        self.lda_ptr.run(n_steps, optimize)

    def get_assignment(self):
        return self.lda_ptr.get_assignment()

    def set_assignment(self, asgn):
        radix = [item for sublist in asgn for item in sublist]
        self.lda_ptr.set_asgn_from_radix(radix)

    def get_radix(self):
        return self.lda_ptr.get_asgn_radix()

    def likelihood(self):
        return self.lda_ptr.likelihood()

    @property
    def n_dk(self):
        cts = self.lda_ptr.get_n_dk()
        return [numpy.array(ct) for ct in cts]

    @property
    def n_k(self):
        cts = self.lda_ptr.get_n_k()
        return cts

    @property
    def n_kw(self):
        cts = self.lda_ptr.get_n_kw()
        return [numpy.array(ct) for ct in cts]

    def topic_word_distributions(self):
        return self.lda_ptr.topic_word_distributions()

    def doc_topic_distributions(self):
        return self.lda_ptr.doc_topic_distributions()


def teach_lda_exact(docs, topics, topics_list=None, alpha=1.0, beta=1.0,
                    return_parts=False):
    """
    Calculates the probability of docs under the teaching model for LDA
    exactly via enumeration of all possible word-topic assignments.

    Notes
    -----
    This is going to be very, very slow. Assume that there are N total word
    in the supplied documents and the there are T topics. The runtime of the
    algorithm is O(T^N).

    Parameters
    ----------
    docs : dict
        List of document data structures (see ldateach.utils.gen_docs)
    topics : list(numpy.ndarray)
        List of n_words-length arrays where each entry topics[t][w] is the
        probability with which word w appears in topic t.
    alpha : float (0, Inf), optional
        Symmetric dirichlet parameter for the distribution of topics in
        documents.
    beta : float (0, Inf), optional
        Symmteric dirichlet parameter for the distribution of words in
        topics.

    Returns
    -------
    logp : double
        The log teaching probability
    """
    n_wrds = len(topics[0])
    n_tpcs = len(topics)
    if not all(len(topic) == n_wrds for topic in topics):
        raise ValueError("Each topic must have the same number of entries")

    if topics_list is None:
        topics_list = [i for i in range(len(topics))]
    else:
        assert type(topics_list) is list
        assert max(topics_list) < len(topics)
        assert min(topics_list) >= 0 

    cdocs = [doc['w'] for doc in docs]
    for cdoc in cdocs:
        if any(w >= n_wrds for w in cdoc):
            raise ValueError('Word index exceeded the max allowed by topics.')
    
    est = ldateach_exact(cdocs, topics, topics_list, alpha, beta)

    if return_parts:
        return est.prior, est.numer, est.denom
    else:
        return est.prior + logsumexp(est.numer) - logsumexp(est.denom)


def teach_lda_unis(docs, topics, topics_list=None, alpha=1.0, beta=1.0,
                   n_samples=1000, seed=None, return_parts=False):
    """
    Approximates the probability of docs under the teaching model for LDA.

    Uses an unbiased Importance Sampling estimate.
    
    Notes
    -----
    The importance function used is a uniform sampling of the word-topic
    assignments, which is just about maximally inefficient.

    Parameters
    ----------
    docs : dict
        List of document data structures (see ldateach.utils.gen_docs)
    topics : list(numpy.ndarray)
        List of n_words-length arrays where each entry topics[t][w] is the
        probability with which word w appears in topic t.
    alpha : float (0, Inf), optional
        Symmetric dirichlet parameter for the distribution of topics in
        documents.
    beta : float (0, Inf), optional
        Symmteric dirichlet parameter for the distribution of words in
        topics.
    n_samples : int
        Number of importance samples to use for the approximation.
    seed : int or None, optional
        The seed for the C++ RNG. If None (default), uses the system time.

    Returns
    -------
    logp : double
        The log teaching probability
    """
    n_wrds = len(topics[0])
    if not all(len(topic) == n_wrds for topic in topics):
        raise ValueError("Each topic must have the same number of entries")

    if topics_list is None:
        topics_list = [i for i in range(len(topics))]
    else:
        assert type(topics_list) is list
        assert max(topics_list) < len(topics)
        assert min(topics_list) >= 0 

    cdocs = [doc['w'] for doc in docs]
    
    if seed is None:
        seed = numpy.random.randint(0, 2**31)

    est = ldateach_unis(cdocs, topics, topics_list, alpha, beta, n_samples,
                        seed)
    if return_parts:
        return est.prior, est.numer, est.denom
    else:
        return est.prior + logsumexp(est.numer) - logsumexp(est.denom)


def evidence_lda_hm(docs, n_topics, n_words, alpha=1.0, beta=1.0,
                    n_samples=100, stop_itr=300, seed=None,
                    return_samples=False):
    """
    Approximates the evidence using the harmonic mean

    Parameters
    ----------
    docs : dict
        List of document data structures (see ldateach.utils.gen_docs)
    n_topics : int
        Number of topics
    n_words : int
        Number of words in vocabulary
    alpha : float (0, Inf), optional
        Symmetric dirichlet parameter for the distribution of topics in
        documents.
    beta : float (0, Inf), optional
        Symmteric dirichlet parameter for the distribution of words in
        topics.
    n_samples : int
        Number of importance samples to use for the approximation.
    stop_itr : int
        Number of LDA Gibbs iterations before a sample is taken
    seed : int or None, optional
        The seed for the C++ RNG. If None (default), uses the system time.

    Returns
    -------
    logp : double
        The evidence
    """
    # dummy lists
    d_tpcs = [numpy.ones(n_words)/n_words for _ in range(n_topics)]
    d_tpclst = [i for i in range(n_topics)]

    nwt = sum(len(d['w']) for d in docs) 

    if seed is not None:
        numpy.random.seed(seed)

    logps = numpy.zeros(n_samples)
    for i in range(n_samples):
        sd = numpy.random.randint(2**31)
        lda = FastLDA(docs, n_topics, n_words, alpha, beta, seed=sd)
        lda.run(n_steps=stop_itr)
        logps[i] = lda.likelihood() 

    if return_samples:
        return -logps
    else:
        return -(logsumexp(-logps) - log(n_samples))


def teach_lda_pgis(docs, topics, topics_list=None, alpha=1.0, beta=1.0,
                   n_samples=1000, seed=None, return_parts=False):
    """
    Approximates the probability of docs under the teaching model for LDA.

    Uses an unbiased Importance Sampling estimate.
    
    Notes
    -----
    The importance function used is a partial Gibbs initalization.

    Parameters
    ----------
    docs : dict
        List of document data structures (see ldateach.utils.gen_docs)
    topics : list(numpy.ndarray)
        List of n_words-length arrays where each entry topics[t][w] is the
        probability with which word w appears in topic t.
    alpha : float (0, Inf), optional
        Symmetric dirichlet parameter for the distribution of topics in
        documents.
    beta : float (0, Inf), optional
        Symmteric dirichlet parameter for the distribution of words in
        topics.
    n_samples : int
        Number of importance samples to use for the approximation.
    seed : int or None, optional
        The seed for the C++ RNG. If None (default), uses the system time.

    Returns
    -------
    logp : double
        The log teaching probability
    """
    n_wrds = len(topics[0])
    if not all(len(topic) == n_wrds for topic in topics):
        raise ValueError("Each topic must have the same number of entries")

    if topics_list is None:
        topics_list = [i for i in range(len(topics))]
    else:
        assert type(topics_list) is list
        assert max(topics_list) < len(topics)
        assert min(topics_list) >= 0 

    cdocs = [doc['w'] for doc in docs]
    
    if seed is None:
        seed = numpy.random.randint(0, 2**31)

    est = ldateach_pgis(cdocs, topics, topics_list, alpha, beta, n_samples,
                        seed)
    if return_parts:
        return est.prior, est.numer, est.denom
    else:
        return est.prior + logsumexp(est.numer) - logsumexp(est.denom)


def _enumlpstep(docsin, idxs, topics, alpha, beta, n_samples, method, seed):
    if method == 'exact':
        logp = teach_lda_exact(docsin, topics, None, alpha, beta)
    elif method == 'unis':
        logp = teach_lda_unis(docsin, topics, None, alpha, beta, n_samples, seed)
    elif method == 'pgis':
        logp = teach_lda_pgis(docsin, topics, None, alpha, beta, n_samples, seed)
    else:
        raise KeyError('Invalid method "{}"'.format(method))

    return logp, idxs


def _enumlpstep_wrp(args):
    return _enumlpstep(*args)


def enumerate_doc_logps(t_docs, docs, topics, alpha=1.0, beta=1.0,
                        method='exact', n_samples=1E4, seed=None):
    """
    Enumerate the logps for every combintaion of t documents.

    Parameters
    ----------
    t_docs : int
        Number of documents to use for teaching.
    docs : dict
        List of document data structures (see ldateach.utils.gen_docs)
    topics : list(numpy.ndarray)
        List of n_words-length arrays where each entry topics[t][w] is the
        probability with which word w appears in topic t.
    alpha : float (0, Inf), optional
        Symmetric dirichlet parameter for the distribution of topics in
        documents.
    beta : float (0, Inf), optional
        Symmteric dirichlet parameter for the distribution of words in
        topics.
    method : {'exact', 'unis', 'pgis'}, optional
        The method to use to caculate (or estimate) the posteriors of the
        document sets.

    Other Parameters
    ----------------
    n_samples : int, optional
        Number of importance samples to use for the approximation. Not used if
        method == 'exact'.
    seed : int or None, optional
        The seed for the C++ RNG. If None (default), uses the system time. Not
        used if method == 'exact'.

    Returns
    -------
    doc_idxs : list
    doc_logps : list
    """

    method = method.lower()
    n_docs = len(docs)
   
    arglist = []
    for idxs in it.combinations(range(n_docs), t_docs): 
        docsin = [docs[idx] for idx in idxs]
        args = (docsin, idxs, topics, alpha, beta, n_samples, method, seed)
        arglist.append(args)

    pool = Pool()
    result = pool.map(_enumlpstep_wrp, arglist)

    doc_idxs = []
    doc_logps = []
    for logp, idxs in result:
        doc_idxs.append(idxs)
        doc_logps.append(logp)

    return doc_idxs, doc_logps

