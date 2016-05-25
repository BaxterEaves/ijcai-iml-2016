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


// compile with $ g++ *.cpp -o test -std=c++11
// run with $ ./test

#include <vector>
#include <map>
#include <set>
#include <random>
#include <cassert>
#include <cmath>
#include "lda.hpp"
#include "dist.hpp"
#include "utils.hpp"

using std::vector;
using std::map;
using std::set;
using std::cout;
using std::endl;
using std::flush;

void vecprint(vector<size_t> v){
    cout << "[";
    for (auto x : v)
        cout << x << " ";
    cout << "\b]\n";
}

int main(){
    std::mt19937 rd;


    std::uniform_real_distribution<double> dist(.1, 2.);
    std::uniform_int_distribution<unsigned int> irand(0, 2>>31);

    vector<size_t> ct_uint = {1, 1, 2};
    vector<double> ct_double = {1, 1, 2};
    double ddlp = dirdisc_lpdf(ct_uint, .5);
    cout << "dirdisc_lpf (size_t) = " << ddlp << endl;
    ddlp = dirdisc_lpdf(ct_double, .5);
    cout << "dirdisc_lpf (double) = " << ddlp << endl;
    cout << "*Should equal -5.75257263882563\n" << endl;


    double ctlp = cat_lpdf(ct_uint, {log(.5), log(.2), log(.3)});
    cout << "cat_lpf (size_t) = " << ctlp << endl;
    ctlp = cat_lpdf(ct_double, {log(.5), log(.2), log(.3)});
    cout << "cat_lpf (double) = " << ctlp << endl;
    cout << "*Should equal -4.71053070164592\n" << endl;


    double lse = logsumexp({log(.5), log(.2), log(.3)});
    cout << "logsumexp = " << lse << endl;
    cout << "should equal 0\n" << endl;

    // ---
    cout << "Binary partition generator check. Should go to 16" << endl;
    vector<size_t> z = {0, 0, 0, 0};
    size_t i = 1;
    while (radix_incr(z, 2)){
        i++;
        cout << i << ") ";
        vecprint(z);
    }

    // ---
    cout << "\nTernary partition generator check. Should go to 27" << endl;
    z = {0, 0, 0};
    i = 1;
    while (radix_incr(z, 3)){
        i++;
        cout << i << ") ";
        vecprint(z);
    }

    // ---
    cout << "\nvec_to_count(3, {0, 1, 2, 0, 1, 2, 0, 1, 2}, 0, 8)\n";
    vecprint(vec_to_count(3, {0, 1, 2, 0, 1, 2, 0, 1, 2}, 0, 8));
    cout << "(Should be [3, 3, 3])\n";

    // ---
    cout << "\nvec_to_count(3, {0, 1, 2, 0, 1, 2, 0, 1, 2}, 2, 7)\n";
    vecprint(vec_to_count(3, {0, 1, 2, 0, 1, 2, 0, 1, 2}, 2, 7));
    cout << "(Should be [2, 2, 2])\n";

    // ---
    std::pair<double, double> ans1 = __ldateach_sngl({{0, 1}, {2, 0}},
            {{.8, .1, .1}, {.2, .3, .5}}, {0, 1}, {0, 1, 1, 1}, 3, .5, .25);

    vector<map<size_t, size_t>> n_dk = {
        {{0, 1}, {1, 1}}, 
        {{1, 2}}
    };
    map<size_t, map<size_t, size_t>> n_kw = {
        {0,
            {{0, 1}}
        },
        {1, 
            {{0, 1}, {1, 1}, {2, 1}}
        }
    };
    map<size_t, size_t> n_k = {{0, 1}, {1, 3}};
    set<size_t> tpcs_ex = {0, 1};
    std::pair<double, double> ans1_sis = __ldateach_sngl_sis({{0, 1}, {2, 0}},
            {{.8, .1, .1}, {.2, .3, .5}}, {0, 1}, {0, 1, 1, 1},
            n_dk, n_kw, n_k, tpcs_ex, .5, .25);

    cout << "\nNumerator: " << ans1.first << " (should be -6.78997224332575)\n";
    cout << "Denominator: " << ans1.second << " (should be -9.60130079388146)\n";
    cout << "Numerator (SIS): " << ans1_sis.first;
    cout << " (should be -6.78997224332575)\n";
    cout << "Denominator (SIS): " << ans1_sis.second;
    cout << " (should be -9.60130079388146)\n";

    // ---
    std::pair<double, double> ans2 = __ldateach_sngl({{0, 1}, {2, 0}},
            {{.8, .1, .1}, {.2, .3, .5}}, {0}, {0, 1, 1, 1}, 3, .5, .25);
    cout << "\nNumerator: " << ans2.first << " (should be -8.72583205652757)\n";
    cout << "Denominator: " << ans2.second << " (should be -9.60130079388146)\n";

    // ---
    LDA lda({{0, 1}, {2, 0}}, 2, 3, .5, .25, 1337);
    lda.set_asgn_from_radix({0, 1, 1, 1});
    cout << "\nLDA likelihood: " << lda.likelihood();
    cout << " (should be -6.5410299991899)" << endl;

    // ---
    Estimator est = ldateach_exact({{0, 1, 2}}, {{.8, .1, .1}, {.2, .3, .5}},
                                   {0, 1}, .5, .5);
    cout << "\nTeaching numer: " << logsumexp(est.numer) << endl; 
    cout << "should equal -3.70908216143146\n" << endl;

    size_t n_times = 100;
    const vector<vector<size_t>> docs = {{1, 1, 1, 1, 1}, {2, 2, 2, 2, 2}};
    const vector<vector<double>> tpcs = {{.05, .8, .05, .05, .05},
                                         {.05, .05, .8, .05, .05},
                                         {.05, .05, .05, .05, .8}};
    double alpha = dist(rd);
    double beta = dist(rd);

    Estimator exact = ldateach_exact(docs, tpcs, {0, 1, 2}, alpha, beta);
    double logp_exact = exact.prior+logsumexp(exact.numer)-logsumexp(exact.denom);
    cout << "ALPHA: " << alpha << ", BETA: " << beta << endl; 
    cout << "EXACT LOGP: " << logp_exact << endl;

    double re_unis = 0;
    double re_pgis = 0;
    for (size_t i=0; i < n_times; ++i){
        cout << i+1 << " " << flush;

        auto sd = irand(rd);
        auto unis = ldateach_unis(docs, tpcs, {0, 1, 2}, alpha, beta, 1000, sd);
        cout << "+" << flush;
        auto pgis = ldateach_pgis(docs, tpcs, {0, 1, 2}, alpha, beta, 1000, sd);
        cout << "-" << flush;

        double lp_unis = unis.prior+logsumexp(unis.numer)-logsumexp(unis.denom);
        double lp_pgis = unis.prior+logsumexp(pgis.numer)-logsumexp(pgis.denom);
        re_unis += std::fabs(1. - lp_unis/logp_exact);
        re_pgis += std::fabs(1. - lp_pgis/logp_exact);

    }
    cout << "RELERR UNIS: " << re_unis/n_times << endl;
    cout << "RELERR PGIS: " << re_pgis/n_times << endl;

    return 0;
}
