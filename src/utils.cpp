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


#include "utils.hpp"

using std::vector;


void inplace_abs_jitter(vector<vector<double>> &m, double amt,
                        std::mt19937 &rng)
{
    std::uniform_real_distribution<double> rand(1-amt, amt+1);
    for(vector<double> &v : m){
        for(double &x : v)
            x += rand(rng);
        double normconst = std::accumulate(v.begin(), v.end(), 0);
        for(double &x : v)
            x /= normconst;
    };
}


vector<vector<double>> vlog(vector<vector<double>> m)
{
    for(auto &r : m)
        std::transform(r.begin(), r.end(), r.begin(), ::log);

    return m;
}


vector<vector<double>> transpose(const vector<vector<double>> &m)
{
    auto n_rows = m.size();
    auto n_cols = m[0].size();

    vector<vector<double>> t(n_cols, vector<double>(n_rows));

    for (size_t i=0; i < n_rows; ++i)
        for (size_t j=0; j < n_cols; ++j)
            t[j][i] = m[i][j];

    return t;
}


vector<vector<double>> lnorm(vector<vector<double>> m)
{
    for(auto &row : m){
        double normconst = logsumexp(row);
        for (auto &val : row){
            val -= normconst;
        }
    } 

    return m;
}


vector<vector<double>> lmult(vector<vector<double>> a, vector<double> b)
{
    for(auto &row : a){
        assert(row.size() == b.size());
        for (size_t i=0; i < row.size(); ++i){
            row[i] += b[i];
        }
    } 

    return a;
}


double logsumexp(const vector<double> &logps)
{
    // if there is a single element in the vector, return that element
    // otherwise there will be log domain problems
    if (logps.size() == 1)
        return logps.front();

    double max = *std::max_element(logps.begin(), logps.end());
    double sum = 0;
    for (size_t i = 0; i < logps.size(); ++i)
        sum += exp(logps[i]-max);

    double retval = log(sum)+max;
    return retval;
}


size_t __do_discrete_draw(const vector<double> &cum_ps, double r)
{
    assert( r <= 1 and r > 0);
    for (size_t i = 0; i < cum_ps.size(); ++i)
        if (r < cum_ps[i])
            return i;

    std::cout << "Failed to find index." << std::endl;
    throw 1;
}


vector<size_t> log_discrete_draw(const vector<double> &logps, size_t n, 
                                 std::mt19937 &rng, bool normed)
{
    auto cum_ps = logps;
    const double normconst = (normed) ? 0 : logsumexp(logps);

    cum_ps[0] = exp(cum_ps[0] - normconst);
    for (size_t i=1; i < cum_ps.size(); i++){
        cum_ps[i] = cum_ps[i-1] + exp(cum_ps[i]-normconst);
    }
    
    std::uniform_real_distribution<double> dist(0, 1);
    vector<size_t> draws(n);
    for (auto &draw : draws){
        double r = dist(rng);
        draw = __do_discrete_draw(cum_ps, r);
    }
    return draws;
}


size_t log_discrete_draw(const vector<double> &logps, std::mt19937 &rng,
                         bool normed)
{
    // normalize
    const double norm_const = (normed) ? 0 : logsumexp(logps);
    std::uniform_real_distribution<double> dist(0, 1);
    double r = dist(rng);
    double cumsum = 0;
    for (size_t i = 0; i < logps.size(); ++i){
        cumsum += exp(logps[i]-norm_const);
        assert(cumsum <= 1. + 10E-8);
        if (r < cumsum)
            return i;
    }

    std::cout << "Failed to find index." << std::endl;
    throw 1;

}


bool radix_incr(vector<size_t> &radix, const size_t base)
{
    const size_t max_elem = base - 1;

    bool carry;
    size_t idx = 0;

    do {
        carry = false;
        ++radix[idx];
        if (radix[idx] > max_elem) {
            radix[idx] = 0;
            carry = true;
        }
        ++idx;
    } while (carry and idx < radix.size());

    return !(idx == radix.size() and carry);
}


vector<size_t> vec_to_count(const size_t n_bins, const vector<size_t> &vec,
                            const size_t start, const size_t end)
{ 
    assert(end < vec.size());

    vector<size_t> count(n_bins);
    for (size_t i=start; i <= end; ++i)
        ++count[vec[i]];
    return count;
}
