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



#ifndef ldateach_utils_GUARD__ 
#define ldateach_utils_GUARD__ 

#include <map>
#include <vector>
#include <cmath>
#include <random>
#include <iostream>
#include <cassert>
#include <algorithm>


void inplace_abs_jitter(std::vector<std::vector<double>> &m, double jitter,
                        std::mt19937 &rng);

std::vector<std::vector<double>> vlog(std::vector<std::vector<double>> m);
std::vector<std::vector<double>> transpose(
        const std::vector<std::vector<double>> &m);
std::vector<std::vector<double>> lnorm(std::vector<std::vector<double>> m);
std::vector<std::vector<double>> lmult(std::vector<std::vector<double>> a,
                                       std::vector<double> b);

double logsumexp(const std::vector<double> &logps);

size_t __do_discrete_draw(const std::vector<double> &cum_logps, double r);
size_t log_discrete_draw(const std::vector<double> &logps, std::mt19937 &rng,
                         bool normed=false);
std::vector<size_t> log_discrete_draw(const std::vector<double> &logps,
                                      size_t n, std::mt19937 &rng,
                                      bool normed=false);

bool radix_incr(std::vector<size_t> &radix, const size_t base);

std::vector<size_t> vec_to_count(const size_t n_bins,
                                 const std::vector<size_t> &vec,
                                 const size_t start, const  size_t end);


template <typename T>
std::pair<size_t, size_t> argminmax(std::vector<T> a)
{
    size_t n = a.size();
    size_t mindex, maxdex;
    T minval, maxval;
    size_t strt = 2;
    
    if (n%2 == 1){
        minval = maxval = a[0];
        mindex = maxdex = 0;
        strt = 1;
    }
    if (n%2 == 0){
        if (a[0] < a[1]){
            minval = a[0];
            maxval = a[1];
            mindex = 0;
            maxdex = 1;
        }else{
            minval = a[1];
            maxval = a[0];
            mindex = 1;
            maxdex = 0;
        }
    }
    for (size_t i=strt; i < a.size(); i += 2){
        if (a[i] < a[i+1]){
            if (minval > a[i]){
                minval = a[i];
                mindex = i;
            }
            if (maxval < a[i+1]){
                maxval = a[i+1];
                maxdex = i+1;
            }
        }else{
            if (minval > a[i+1]){
                minval = a[i+1];
                mindex = i+1;
            }
            if (maxval < a[i]){
                maxval = a[i];
                maxdex = i;
            }
        }
    }
    return {mindex, maxdex};
};   

class Counter {
    std::map<size_t, size_t> _counts;
    size_t _n = 0;

public:
    Counter(std::vector<size_t> xs){
        for(auto x: xs){
            add(x);
        }
    };
    void add(size_t x){
        ++_counts[x];
        ++_n;
    };
    void subtract(size_t x){
        size_t &ct = _counts.at(x);
        if (ct == 1){
            _counts.erase(x);
        }else{
            --ct;
        }
        --_n;
    };
    size_t rand_elem(std::mt19937 &rng){
        std::uniform_int_distribution<size_t> dist(1, _n);
        size_t r = dist(rng);
        size_t sum = 0;
        for (auto &kv : _counts){
            sum += kv.second;
            if(r <= sum){
                return kv.first;
            }
        }
        throw std::out_of_range ("Cound not draw random element.");
    };
    void print(){
        std::cout << "(" << _n << "){ ";
        for (auto kv : _counts){
            std::cout << kv.first << ":" << kv.second << " ";
        }
        std::cout << "}" << std::endl;
    }
};


class AliasTable_LDA {
    std::vector<size_t> _k;
    std::vector<double> _v;
    std::vector<double> _p;
    size_t _n_entries;
    size_t _samples_extracted;
    size_t _max_samples;
    
    double _beta;
    double _betaW;
    size_t _wrd;
    const std::vector<std::vector<size_t>> &_n_kw;
    const std::vector<size_t> &_n_k;
    
    std::uniform_real_distribution<double> urand;
    
public:
    AliasTable_LDA(const std::vector<std::vector<size_t>> &n_kw,
                   const std::vector<size_t> &n_k, double beta,
                   size_t wrd, size_t max_samples) :
    _n_kw(n_kw), _n_k(n_k), urand(std::uniform_real_distribution<double>(0, 1))
    {
        _n_entries = n_k.size();
        _samples_extracted = 0;
        
        _p.resize(_n_entries, 0);
        _k.resize(_n_entries, 0);
        _v.resize(_n_entries, 0);
        
        _beta = beta;
        _wrd = wrd;
        _betaW = beta*n_kw[0].size();
        _max_samples = max_samples;
        
        rebuild();
    };
   
    double getp(size_t idx){
        return _p[idx];
    }

    void rebuild(){
        double a = 1/static_cast<double>(_n_entries);
        
        for (size_t k=0; k < _n_entries; ++k){
            _p[k] = static_cast<double>(_n_kw[k][_wrd]) + _beta;
            _p[k] /= static_cast<double>(_n_k[k]) + _betaW;
        }
        
        std::vector<double> p = _p;
        
        std::iota(_k.begin(), _k.end(), 0);
        for (size_t i=0; i < _n_entries; ++i){
            _v[i] = static_cast<double>(i+i)*a;
        }
        
        for (size_t ii=0; ii < _n_entries-1; ++ii){
            auto aminmax = argminmax(p);
            size_t i = aminmax.first;
            size_t j = aminmax.second;
            
            _k[i] = j;
            _v[i] = i*a + p[i];
            
            p[j] -= (a-p[i]);
            p[i] = a;
        }
        _samples_extracted = 0;
    };
    
    std::pair<size_t, double> draw(std::mt19937 &rng){
        if (_samples_extracted >= _max_samples){
            rebuild();
        }
        double u = urand(rng);
        ++_samples_extracted;
        size_t j = _n_entries*u;
        size_t index =  (u < _v[j]) ? j : _k[j];
        double p = _p[index];

        if(index >= _n_entries){
            std::cout << "u: " << u << std::endl;
            std::cout << "j: " << j << std::endl;
            std::cout << "_v[j]: " << _v[j] << std::endl;
            std::cout << "_k[j]: " << _k[j] << std::endl;
            std::cout << "_n_entries: " << _n_entries << std::endl;
        }

        return {index, p};
    };
};
#endif
