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


import random
import numpy as np

from math import log
from ldateach.utils import discrete_draw
from scipy.stats import dirichlet
from scipy.special import gammaln
from collections import Counter

WORD = 0
DOC = 1


class LDA(object):
    """ Latent Dirichlet Allocation (LDA) via collapsed Gibbs sampling.

    Attributes
    ----------
    z : list<numpy.ndarray>
        Entry z[d][w] is the topic to which the w^th word in document d is
        assigned
    n_dk : numpy.ndarrary(n_docs, n_topics)
        number of words assigned to topic k in doc d
    """
    def __init__(self, docs, n_topics, n_words, alpha=1.0, beta=1.0,
                 init_mode='prior', seed=None):
        """
        Parameters
        ----------
        docs : list<dict>
            list of document data structures
        n_topics : int
            number of topics
        n_words : int
            number of words in corpus (vocabulary)
        alpha : float (0, Inf), optional
            symmetric Dirchlet parameter for word/document distribution
        beta : float (0, Inf), optional
            symmetric Drichlet parameter for topic/document distribution
        """
        self._docs = docs
        self._n_docs = len(docs)
        self._n_topics = n_topics
        self._n_words = n_words
        self._alpha = alpha
        self._beta = beta

        # number of words assigned to topic k in doc d
        self._n_dk = np.zeros((self._n_docs, self._n_topics,))
        # number of times word w is assigned to topic k
        self._n_kw = np.zeros((self._n_topics, self._n_words,))
        # number of times any word is assigned to topic k
        self._n_k = np.zeros((1, self._n_topics,))
        # Entry z[d][w] is the topic to which the w^th word in document d is
        # assigned
        self._z = []
        self._key = []

        for d, doc in enumerate(self._docs):
            self._z.append([])
            if init_mode == 'prior':
                theta_k = dirichlet.rvs([self._alpha]*self._n_topics)[0]
            elif init_mode == 'random':
                theta_k = np.ones(self._n_topics)/self._n_topics
            else:
                raise ValueError("init_mode must be 'random' or 'prior'")
            for w, wrd in enumerate(doc['w']):
                topic = int(discrete_draw(theta_k))
                self._z[d].append(topic)

                self._n_dk[d, topic] += 1.0
                self._n_kw[topic, wrd] += 1.0
                self._n_k[0, topic] += 1.0
                self._key.append((d, wrd, w,))

    def get_assignment(self):
        return self._z

    def set_assignment(self, asgn):
        for d, doc in enumerate(self._docs):
            for w, wrd in enumerate(doc['w']):
                tpc_old = self._z[d][w]
                tpc_new = asgn[d][w]

                self._n_dk[d, tpc_old] -= 1.0
                self._n_kw[tpc_old, wrd] -= 1.0
                self._n_k[0, tpc_old] -= 1.0

                self._n_dk[d, tpc_new] += 1.0
                self._n_kw[tpc_new, wrd] += 1.0
                self._n_k[0, tpc_new] += 1.0

                self._z[d][w] = tpc_new

    def get_radix(self):
        return [item for sublist in self._z for item in sublist]

    def doc_topic_distributions(self):
        theta = np.copy(self._n_dk) + self._alpha
        theta /= np.sum(theta, axis=1)[:, np.newaxis]
        return theta

    def topic_word_distributions(self):
        phi = np.copy(self._n_kw) + self._beta
        phi /= np.sum(phi, axis=1)[:, np.newaxis]
        return phi

    @property
    def n_dk(self):
        return np.copy(self._n_dk)

    @property
    def n_kw(self):
        return np.copy(self._n_kw)

    def log_conditional(self, d, k, word):
        logp = log(self._n_dk[d, k] + self._alpha)
        logp += log(self._n_kw[k, word] + self._beta)
        logp -= log(self._n_k[0, k] + self._beta*self._n_words)
        return logp

    def likelihood(self):
        ll = self._n_topics*gammaln(self._n_words*self._beta)
        ll -= (self._n_words*self._n_topics)*gammaln(self._beta)
        for k in range(self._n_topics):
            ll -= gammaln(self._n_k[0, k] + self._n_words*self._beta)
            for w in range(self._n_words):
                ll += gammaln(self._n_kw[k, w] + self._beta)
        return ll

    def prior(self):
        lp = self._n_docs*gammaln(self._n_topics*self._alpha)
        lp -= (self._n_docs*self._n_topics)*gammaln(self._alpha)
        for d in range(self._n_docs):
            lp -= gammaln(len(self._docs[d]['w']) + self._n_topics*self._alpha)
            for t in range(self._n_topics):
                lp += gammaln(self._n_dk[d, t] + self._alpha)
        return lp

    def step(self):
        for d, word, w in self._key:
            topic = self._z[d][w]

            self._n_dk[d, topic] -= 1.0
            self._n_kw[topic, word] -= 1.0
            self._n_k[0, topic] -= 1.0

            logp_k = np.zeros(self._n_topics)
            for k in range(self._n_topics):
                logp_k[k] = self.log_conditional(d, k, word)

            topic = discrete_draw(logp_k, logp=True)
            self._z[d][w] = topic
            self._n_dk[d, topic] += 1.0
            self._n_kw[topic, word] += 1.0
            self._n_k[0, topic] += 1.0

    def run(self, n_steps=1):
        """ Run the sampler for n steps """
        for _ in range(n_steps):
            random.shuffle(self._key)
            self.step()


class LDA_kt(LDA):
    """ Gibbs sampler for LDA with known topics """
    def __init__(self, docs, phis, n_words, alpha=1.0, beta=1.0):
        """
        Parameters
        ----------
        docs : list<dict>
            list of document data structures
        phis : list(numpy.ndarray)
            List of topics. A topic is an n_words-length vector where each
            entry, w, is the probabiliyt of word w under the topic.
        n_words : int
            number of words in corpus (vocabulary)
        alpha : float (0, Inf), optional
            symmetric Dirchlet parameter for word/document distribution
        beta : float (0, Inf), optional
            symmetric Drichlet parameter for topic/document distribution
        """
        n_topics = len(phis)

        self.phis = phis
        super(LDA_kt, self).__init__(docs, n_topics, n_words, alpha, beta)

    def log_conditional(self, d, k, word):
        logp = log(self.phis[k][word]) + log(self._n_dk[d, k] + self._alpha)
        return logp


class SparseLDA(LDA):
    def __init__(self, docs, n_topics, n_words, alpha, beta, seed=None):
        super(SparseLDA, self).__init__(docs, n_topics, n_words, alpha, beta,
                                        seed=seed)
        self._cnst = self._beta*self._n_words + self._n_k[0, :]
        self._s = self._alpha*self._beta*np.sum(1./self._cnst)

        self._q_cache = (self._alpha + self.n_dk)/self._cnst

        # store existing topics for docs and words
        self._doc_topics = [set() for _ in range(self._n_docs)]
        self._word_topics = [set() for _ in range(self._n_words)]
        for d, doc in enumerate(self._docs):
            for w, word, in enumerate(doc['w']):
                t = self._z[d][w]
                self._doc_topics[d].add(t)
                self._word_topics[word].add(t)

    def run(self, n_steps=1):
        for d, doc in enumerate(self._docs):
            self._r = 0
            for t in self._doc_topics[d]:
                self._r += self.n_dk[d, t]*self._beta/self._cnst[t]
            for w, _ in enumerate(doc['w']):
                self.step(d, w)

    def _sub_sr(self, d, t):
        # subtract from s and r
        self._s -= self._alpha*self._beta/self._cnst[t]
        self._r -= self._beta*self._n_dk[d, t]/self._cnst[t]

    def _add_sr(self, d, t):
        # subtract from s and r
        self._s += self._alpha*self._beta/self._cnst[t]
        self._r += self._beta*self._n_dk[d, t]/self._cnst[t]

    def _update_q_cache(self, d, t):
        self._q_cache[d, t] = (self._alpha + self._n_dk[d, t])/self._cnst[t]

    def _remove_word(self, d, w):
        t = self._z[d][w]
        word = self._docs[d]['w'][w]

        # mark
        self._z[d][w] = -1

        # subtract from s and r
        self._sub_sr(d, t)

        # remove from counts
        self._n_kw[t, word] -= 1
        self._n_dk[d, t] -= 1
        self._n_k[0, t] -= 1

        # update constant
        self._cnst[t] = self._beta*self._n_words + self._n_k[0, t]

        # add back into s and r and update q_cache
        self._add_sr(d, t)
        self._update_q_cache(d, t)

        if self._n_kw[t, word] == 0:
            self._word_topics[word].remove(t)
        if self._n_dk[d, t] == 0:
            self._doc_topics[d].remove(t)

    def _insert_word(self, d, w, t):
        assert self._z[d][w] == -1
        word = self._docs[d]['w'][w]

        # set
        self._z[d][w] = t

        # subtract from s and r
        self._sub_sr(d, t)

        # insert into counts
        self._n_kw[t, word] += 1
        self._n_dk[d, t] += 1
        self._n_k[0, t] += 1

        # update constant
        self._cnst[t] = self._beta*self._n_words + self._n_k[0, t]

        # add back into s and r and update q_cache
        self._add_sr(d, t)
        self._update_q_cache(d, t)

        if self._n_kw[t, word] == 1:
            self._word_topics[word].add(t)
        if self._n_dk[d, t] == 1:
            self._doc_topics[d].add(t)

    def set_assignment(self, asgn):
        super(SparseLDA, self).set_assignment(asgn)
        self._cnst = self._beta*self._n_words + self._n_k[0, :]
        self._s = self._alpha*self._beta*np.sum(1./self._cnst)

        self._q_cache = (self._alpha + self.n_dk)/self._cnst

        # store existing topics for docs and words
        self._doc_topics = [set() for _ in range(self._n_docs)]
        self._word_topics = [set() for _ in range(self._n_words)]
        for d, doc in enumerate(self._docs):
            for w, word, in enumerate(doc['w']):
                t = self._z[d][w]
                self._doc_topics[d].add(t)
                self._word_topics[word].add(t)

    def step(self, d, w):
        word = self._docs[d]['w'][w]

        self._remove_word(d, w)

        q = 0
        for k in self._word_topics[word]:
            q += self._q_cache[d, k]*self._n_kw[k, word]

        u = random.random()*(self._s + self._r + q)

        t = None
        if u < self._s:
            # uniform
            t = discrete_draw(-np.log(self._cnst), logp=True)

        elif u < self._s + self._r:
            # document
            ks = [k for k in self._doc_topics[d]]
            logps = np.zeros(len(self._doc_topics[d]))
            for i, k in enumerate(self._doc_topics[d]):
                logps[i] = log(self._n_dk[d, k]) + log(self._beta/self._cnst[k])

            t = ks[discrete_draw(logps, logp=True)]

        else:
            # word
            ks = [k for k in self._word_topics[word]]
            logps = np.zeros(len(self._word_topics[word]))
            for i, k in enumerate(self._word_topics[word]):
                logps[i] = log(self._n_kw[k, word]) + log(self._q_cache[d, k])

            t = ks[discrete_draw(logps, logp=True)]

        self._insert_word(d, w, t)


class LazyLDA(LDA):
    """ O(1) LDA sampler """
    def __init__(self, docs, n_topics, n_words, alpha, beta, init_mode='prior',
                 seed=None):
        super(LazyLDA, self).__init__(docs, n_topics, n_words, alpha, beta,
                                      init_mode=init_mode, seed=seed)
        self._proposal_type = WORD
        self._n = sum(len(doc) for doc in docs)
        self._n_w = np.zeros(n_words)
        self._z_w = dict()

        for d, doc in enumerate(self._docs):
            for w, wrd in enumerate(doc['w']):
                topic = self._z[d][w]
                self._n_w[wrd] += 1
                if not self._z_w.get(wrd):
                    self._z_w[wrd] = Counter()
                self._z_w[wrd].update([topic])

    def step(self, d, w):
        s = self._z[d][w]
        wrd = self._docs[d]['w'][w]

        self._n_k[0, s] -= 1
        self._n_kw[s, wrd] -= 1
        self._n_dk[d, s] -= 1

        if self._proposal_type is WORD:
            p = self._n_w[wrd]-1
            p /= p + self._beta*self._n_topics*self._n_words
            if random.random() < p:
                r = random.randrange(self._n_w[wrd]-1)
                sm = 0
                for t, ct in self._z_w[wrd].most_common():
                    sm += ct
                    if t == s:
                        sm -= 1
                    if r < sm:
                        break
            else:
                t = random.randrange(self._n_topics)

            q = log(self._n_kw[s, wrd] + self._beta*self._n_words)
            q -= log(self._n_kw[t, wrd] + self._beta*self._n_words)

            π = log(self._n_dk[d, t] + self._alpha)
            π -= log(self._n_dk[d, s] + self._alpha)
            π += q

        else:
            l = float(len(self._docs[d]['w'])-1.)
            p = l / (l + self._alpha*self._n_topics)
            if random.random() < p:
                self._z[d][w] = -1
                t = random.choice(self._z[d])
                while t == -1:
                    t = random.choice(self._z[d])
            else:
                t = random.randrange(self._n_topics)
            π = 0

        π += log(self._n_kw[t, wrd] + self._beta)
        π -= log(self._n_kw[s, wrd] + self._beta)
        π += log(self._n_k[0, s] + self._beta*self._n_words)
        π -= log(self._n_k[0, t] + self._beta*self._n_words)

        if log(random.random()) < π:
            self._z_w[wrd].subtract([s])
            self._z_w[wrd].update([t])
            s = t

        self._z[d][w] = s
        self._n_k[0, s] += 1
        self._n_kw[s, wrd] += 1
        self._n_dk[d, s] += 1

    def set_assignment(self, asgn):
        for d, doc in enumerate(self._docs):
            for w, wrd in enumerate(doc['w']):
                tpc_old = self._z[d][w]
                tpc_new = asgn[d][w]

                self._n_dk[d, tpc_old] -= 1.0
                self._n_kw[tpc_old, wrd] -= 1.0
                self._n_k[0, tpc_old] -= 1.0

                self._n_dk[d, tpc_new] += 1.0
                self._n_kw[tpc_new, wrd] += 1.0
                self._n_k[0, tpc_new] += 1.0

                self._z_w[wrd].subtract([tpc_old])
                self._z_w[wrd].update([tpc_new])

                self._z[d][w] = tpc_new

    def run(self, n_sweeps=1):
        for _ in range(n_sweeps):
            random.shuffle(self._key)
            for d, _, w in self._key:
                self._proposal_type = WORD
                self.step(d, w)
                self._proposal_type = DOC
                self.step(d, w)
