// ldateach, generate/choose documents to teach topics model
// Copyright (C) 2016 Baxter S. Eaves Jr.

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.


#ifndef ldateach_dist_GUARD__ 
#define ldateach_dist_GUARD__ 

#include <map>
#include <cmath>
#include <vector>
#include <random>
#include <numeric>
#include <cassert>
#include <iostream>


double mul_lpdf(const std::vector<size_t> &cts, const std::vector<double> &p);
double cat_lpdf(const std::vector<size_t> &cts, const std::vector<double> &p);
double cat_lpdf(const std::vector<double> &cts, const std::vector<double> &p);
std::vector<double> multdirpred(const std::vector<size_t> &cts, double conc);
double dir_lpdf(const std::vector<double> &weights, double alpha);
double dirdisc_lpdf(const std::vector<size_t> &cts, double alpha);
double dirdisc_lpdf(const std::vector<double> &cts, double alpha);
double dirdisc_lpdf_sparse(const std::map<size_t, size_t> &cts, size_t k,
                           double alpha);
double dirmul_lpdf(const std::vector<size_t> &cts, double alpha);
std::vector<double> dir_rand(const double alpha, const size_t k,
                             std::mt19937 &rng);

#endif
