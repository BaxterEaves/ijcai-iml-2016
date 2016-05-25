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


#include "dist.hpp"

using std::vector;
using std::map;
using std::accumulate;


// --- Categorical (discrete) pdf for uint and double counts
double cat_lpdf(const vector<size_t> &cts, const vector<double> &logp)
{
    assert(cts.size() == logp.size());
    double lp = 0;
    for (size_t k=0; k < cts.size(); ++k){
        double xk = static_cast<double>(cts[k]);
        lp += xk*logp[k];
    }
    return lp;
}


double cat_lpdf(const vector<double> &cts, const vector<double> &logp)
{
    assert(cts.size() == logp.size());
    double lp = 0;
    for (size_t k=0; k < cts.size(); ++k){
        lp += cts[k]*logp[k];
    }
    return lp;
}


// ---
double mul_lpdf(const vector<size_t> &cts, const vector<double> &logp)
{
    double lp = cat_lpdf(cts, logp);

    double n = 0;
    for (auto &xk : cts){
        n += xk;
        lp -= lgamma(xk + 1);
    }
    lp += lgamma(n + 1);
    return lp; 
}


// ---
vector<double> multdirpred(const vector<size_t> &cts, double conc)
{
    vector<double> dist(cts.size());

    double sum = 0;
    for (const auto &ct : cts)
        sum += (static_cast<double>(ct) + conc);

    for (size_t i=0; i < cts.size(); ++i)
        dist[i] = (static_cast<double>(cts[i]) + conc)/sum;

    return dist;
}


// ---
std::vector<double> dir_rand(const double alpha, const size_t k,
                             std::mt19937 &rng)
{
    std::gamma_distribution<double> dist(alpha, 1.0);
    std::vector<double> x(k, 0);
    double sum = 0;

    for (size_t i=0; i < k; ++i){
        double g = dist(rng);
        x[i] = g;
        sum += g;
    }

    for (auto &y : x)
        y /= sum;

    return x;
}


// ---
double dir_lpdf(const vector<double> &weights, double alpha)
{
    double k = static_cast<double>(weights.size());
    double sum_a = alpha*k;

    double prd_wghts = 0;
    for (size_t w=0; w < weights.size(); w++)
        prd_wghts += log(weights[w])*(alpha-1.0);

    return lgamma(sum_a) - k*lgamma(alpha) + prd_wghts;
}


// --- Dirichlet-discrete distribution for uint and double counts
double dirdisc_lpdf(const vector<size_t> &cts, double alpha)
{
    double n = static_cast<double>(accumulate(cts.begin(), cts.end(), 0));
    double k = static_cast<double>(cts.size());
    double sum_a = alpha*k;

    double prd_xa = 0;
    for (const auto &ct : cts)
        prd_xa += lgamma(static_cast<double>(ct) + alpha);

    return lgamma(sum_a) - lgamma(n+sum_a) + prd_xa - k*lgamma(alpha);
}


double dirdisc_lpdf(const vector<double> &cts, double alpha)
{
    double n = accumulate(cts.begin(), cts.end(), 0);
    double k = static_cast<double>(cts.size());
    double sum_a = alpha*k;

    double prd_xa = 0;
    for (const auto &ct : cts)
        prd_xa += lgamma(ct + alpha);

    return lgamma(sum_a) - lgamma(n+sum_a) + prd_xa - k*lgamma(alpha);
}


// --- Dirichlet-discrete distribution for sparse vectors 
double dirdisc_lpdf_sparse(const map<size_t, size_t> &cts, size_t k, double alpha)
{
    double n = 0; 
    double sum_a = alpha*k;

    double prd_xa = 0;
    for (const auto &wct : cts){
        double ct = static_cast<double>(wct.second);
        prd_xa += lgamma(ct + alpha);
        n += ct;
    }
    prd_xa += lgamma(alpha)*(k-cts.size());

    return lgamma(sum_a) - lgamma(n+sum_a) + prd_xa - k*lgamma(alpha);
}


// ---
double dirmul_lpdf(const vector<size_t> &cts, double alpha)
{
    double n = static_cast<double>(accumulate(cts.begin(), cts.end(), 0));
    double k = static_cast<double>(cts.size());
    double sum_a = alpha*k;

    double prd_xa = 0;
    double prd_x1 = 0;
    for (const auto &ct : cts){
        prd_xa += lgamma(ct + alpha);
        prd_x1 += lgamma(ct + 1.);
    }

    return lgamma(sum_a) + lgamma(n+1) - lgamma(n+sum_a) - prd_x1 +\
        prd_xa - k*lgamma(alpha);
}
