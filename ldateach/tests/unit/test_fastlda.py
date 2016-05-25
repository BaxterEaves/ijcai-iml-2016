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
import numpy

from ldateach.fastlda import FastLDA

FLOAT_ERROR_TOL = 10E-6


# --- FIXTURES
@pytest.fixture
def docset():
    docs = [{'w': [1, 0, 3, 2]}, {'w': [2, 0, 0, 3]}]
    n_topics = 3
    n_words = 4
    alpha = .1
    beta = .3
    return docs, n_topics, n_words, alpha, beta


@pytest.fixture
def lda():
    docs = [{'w': [1, 0, 3, 2]}, {'w': [2, 0, 0, 3]}]
    return FastLDA(docs, 3, 4, .1, .3)


# --- BEGIN TESTS
def test_init_smoke(docset):
    docs, n_topics, n_words, alpha, beta = docset
    FastLDA(docs, n_topics, n_words, alpha, beta)


def test_run_smoke(docset):
    docs, n_topics, n_words, alpha, beta = docset
    lda = FastLDA(docs, n_topics, n_words, alpha, beta)
    lda.run(10)


def test_get_assignment(lda):
    asgn = lda.get_assignment()

    assert isinstance(asgn, list)
    assert len(asgn) == 2  # number of documents
    assert isinstance(asgn[0], list)
    assert len(asgn[0]) == 4  # number of words in doc 0
    assert len(asgn[1]) == 4  # number of words in doc 1
    assert isinstance(asgn[0][0], int)


def test_get_radix(lda):
    radix = lda.get_radix()

    assert isinstance(radix, list)
    assert len(radix) == 8
    assert isinstance(radix[0], int)

    asgnr = [item for sublist in lda.get_assignment() for item in sublist]

    # radix should just be a flattened assignment
    assert all(zi == zj for zi, zj in zip(radix, asgnr))


def test_set_assignment(lda):
    asgn = [[0, 1, 2, 1], [0, 1, 0, 2]]
    lda.set_assignment(asgn)

    radix = lda.get_radix()
    assert all(i == j for i, j in zip(radix, [0, 1, 2, 1, 0, 1, 0, 2]))


def test_likelihood_smoke(lda):
    assert isinstance(lda.likelihood(), float)


def test_n_dk(lda):
    n_dk = lda.n_dk

    assert isinstance(n_dk, list)
    assert len(n_dk) == 2  # number of documents
    assert isinstance(n_dk[0], numpy.ndarray)
    assert isinstance(n_dk[1], numpy.ndarray)
    assert len(n_dk[0]) == 3  # number of topics
    assert len(n_dk[1]) == 3


def test_n_k(lda):
    n_k = lda.n_k

    assert isinstance(n_k, list)
    assert isinstance(n_k[0], int)
    assert sum(n_k) == 8  # total number of words


def test_n_kw(lda):
    n_kw = lda.n_kw

    assert isinstance(n_kw, list)
    assert len(n_kw) == 3  # number of topics

    assert all(isinstance(n_w, numpy.ndarray) for n_w in n_kw)
    assert all(len(n_w) == 4 for n_w in n_kw)  # number of words


def test_topic_word_distributions(lda):
    phi = lda.topic_word_distributions()

    assert isinstance(phi, list)
    assert len(phi) == 3  # number of topics
    assert all(isinstance(phi_k, list) for phi_k in phi)
    assert all(len(phi_k) == 4 for phi_k in phi)  # number of words
    assert all(abs(sum(phi_k)-1.) < FLOAT_ERROR_TOL for phi_k in phi)


def test_doc_topic_distributions(lda):
    theta = lda.doc_topic_distributions()

    assert isinstance(theta, list)
    assert len(theta) == 2  # number of docs
    assert all(isinstance(theta_k, list) for theta_k in theta)
    assert all(len(theta_k) == 3 for theta_k in theta)  # number of topics
    assert all(abs(sum(theta_k)-1.) < FLOAT_ERROR_TOL for theta_k in theta)


def test_seed_constency(docset):
    docs, n_topics, n_words, alpha, beta = docset

    lda = FastLDA(docs, n_topics, n_words, alpha, beta, 1234)
    z1_prerun = lda.get_radix()
    lda.run(10)
    z1_postrun = lda.get_radix()

    assert not all(i == j for i, j in zip(z1_prerun, z1_postrun))

    lda = FastLDA(docs, n_topics, n_words, alpha, beta, 1234)
    z2_prerun = lda.get_radix()

    assert all(i == j for i, j in zip(z1_prerun, z2_prerun))

    lda.run(10)
    z2_postrun = lda.get_radix()

    assert all(i == j for i, j in zip(z2_postrun, z2_postrun))

    lda = FastLDA(docs, n_topics, n_words, alpha, beta, 1337)
    z3_prerun = lda.get_radix()

    assert not all(i == j for i, j in zip(z1_prerun, z3_prerun))


def test_multiple_step(docset):
    docs, n_topics, n_words, alpha, beta = docset

    lda1 = FastLDA(docs, n_topics, n_words, alpha, beta, 1234)
    lda1.run(1)
    z1 = lda1.get_radix()

    lda2 = FastLDA(docs, n_topics, n_words, alpha, beta, 1234)
    lda2.run(2)
    z2 = lda2.get_radix()

    assert not all(i == j for i, j in zip(z1, z2))
