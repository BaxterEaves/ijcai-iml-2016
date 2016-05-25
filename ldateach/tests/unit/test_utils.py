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
import numpy as np

from ldateach import utils

FLOAT_ERROR_TOL = 10E-8
N_DRAWS = 100000


def binerr(x, expect):
    return np.sum(np.abs(1-np.bincount(x, minlength=len(expect))/expect))


# -- Discrete draw tests
def test_discrete_draw_exp_uniform():
    u = np.ones(2)/2.
    x = utils.discrete_draw(u, n=N_DRAWS)

    assert len(x) == N_DRAWS
    assert not np.any(np.isnan(x))
    assert binerr(x, u*N_DRAWS) < .05


def test_discrete_draw_exp_nonuniform():
    u = np.array([.2, .8])
    x = utils.discrete_draw(u, n=N_DRAWS)

    assert len(x) == N_DRAWS
    assert not np.any(np.isnan(x))
    assert binerr(x, u*N_DRAWS) < .05


def test_discrete_draw_exp_point():
    u = np.array([0., 1.])
    x = utils.discrete_draw(u, n=N_DRAWS)

    assert len(x) == N_DRAWS
    assert not np.any(np.isnan(x))
    assert not np.any(x == 0)


def test_discrete_draw_log_uniform():
    u = np.ones(2)/2.
    x = utils.discrete_draw(np.log(u), n=N_DRAWS, logp=True)

    assert len(x) == N_DRAWS
    assert not np.any(np.isnan(x))
    assert binerr(x, u*N_DRAWS) < .05


def test_discrete_draw_log_nonuniform():
    u = np.array([.2, .8])
    x = utils.discrete_draw(np.log(u), n=N_DRAWS, logp=True)

    assert len(x) == N_DRAWS
    assert not np.any(np.isnan(x))
    assert binerr(x, u*N_DRAWS) < .05


def test_discrete_draw_log_point():
    n_samples = 1000
    u = np.array([0, 1])
    x = utils.discrete_draw(np.log(u), n=n_samples, logp=True)

    assert len(x) == n_samples
    assert not np.any(x == 0)
    assert not np.any(np.isnan(x))


# -- Dirichlet-discrete log PDF tests
def test_log_dirdisc_pdf_values():
    # XXX: Values taken from a matlab program
    f = utils.log_dirdisc_pdf(np.array([1., 1., 1.]), 1.)
    v = -4.0943445622221
    assert abs(f-v) < FLOAT_ERROR_TOL

    f = utils.log_dirdisc_pdf(np.array([1., 2., 1.]), .5)
    v = -5.75257263882563
    assert abs(f-v) < FLOAT_ERROR_TOL

    f = utils.log_dirdisc_pdf(np.array([1., 2., 1.]), 2.)
    v = -4.83628190695148
    assert abs(f-v) < FLOAT_ERROR_TOL


def test_log_dirdisc_pdf_symmetry():

    f1 = utils.log_dirdisc_pdf(np.array([1., 3., 2.]), .5)
    f2 = utils.log_dirdisc_pdf(np.array([1., 2., 3.]), .5)
    f3 = utils.log_dirdisc_pdf(np.array([3., 2., 1.]), .5)

    assert f1 == f2
    assert f2 == f3


def test_log_dirdisc_pdf_empty_count():
    f = utils.log_dirdisc_pdf(np.array([0, 0, 0]), .5)
    assert abs(f-0) < FLOAT_ERROR_TOL


# --- Discrete distribution log PDF
def test_log_discrete_pdf_values():
    f = utils.log_discrete_pdf(np.array([1, 1, 1]), np.ones(3)/3)
    v = -3.29583686600433
    assert abs(f-v) < FLOAT_ERROR_TOL

    f = utils.log_discrete_pdf(np.array([1, 1, 2]), np.ones(3)/3)
    v = -4.39444915467244
    assert abs(f-v) < FLOAT_ERROR_TOL

    f = utils.log_discrete_pdf(np.array([1, 1, 2]), np.array([.2, .3, .5]))
    v = -4.19970507787993
    assert abs(f-v) < FLOAT_ERROR_TOL


# --- Cosine distance
def tes_cosdist_qualities():
    a = np.array([1, 2, 3])
    assert utils.cosdist(a, a) == 0.0

    distrev = utils.cosdist(a, a[::-1])

    assert distrev > 0.0
    assert distrev > utils.cosdist(a, np.array([2, 3, 1]))
