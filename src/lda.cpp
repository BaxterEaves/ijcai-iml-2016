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


#include "lda.hpp"

using std::vector;
using std::map;
using std::set;
using std::pair;
using std::tie;
using std::ignore;
using std::transform;


// -- LightLDA
LightLDA::LightLDA(vector<vector<size_t>> docs, size_t n_tpcs, size_t n_wrds,
                   double alpha, double beta, int seed) :
    LDA(docs, n_tpcs, n_wrds, alpha, beta, seed)
{
    for (size_t wrd=0; wrd < _n_wrds; ++wrd){
        _alias_tables.emplace_back(_n_kw, _n_k, _beta, wrd, _n_tpcs);
    }
}

double LightLDA::step(Concordance dw, double temp)
{
    const auto didx = dw.doc_idx;
    const auto wrd = dw.word;
    const auto widx = dw.word_idx;
    const auto t_old = _asgmnt[didx][widx];

    std::uniform_real_distribution<double> urand(0.0, 1.0);

    --_n_k[t_old];
    --_n_kw[t_old][wrd];
    --_n_dk[didx][t_old];

    size_t t_new;
    double pi = 0;
    if(_word_prop){
        auto draw = _alias_tables[wrd].draw(_rng);
        t_new = draw.first;

        assert(t_new < _n_tpcs);

        if (t_old != t_new){
            double p_new = draw.second;
            double p_old = _alias_tables[wrd].getp(t_old);
            pi -= p_new;
            pi += p_old;
            pi += log(_n_dk[didx][t_new] + _alpha);
            pi -= log(_n_dk[didx][t_old] + _alpha);
        }
    }else{
        double l = static_cast<double>(_docs[didx].size()-1);
        double p = l / (l + _alpha*_n_tpcs); 
        if (urand(_rng) < p){
            std::uniform_int_distribution<size_t> uint(0, _docs[didx].size()-1);
            size_t tnidx = 0;
            do{
                tnidx = uint(_rng); 
            }while(tnidx == widx);
            t_new = _asgmnt[didx][tnidx];
        }else{
            std::uniform_int_distribution<size_t> uint(0, _n_tpcs-1);
            t_new = uint(_rng); 
        }
    }

    if (t_old != t_new){
        pi += log(_n_kw[t_new][wrd] + _beta);
        pi -= log(_n_kw[t_old][wrd] + _beta);

        pi += log(_n_k[t_old] + _beta*_n_wrds);
        pi -= log(_n_k[t_new] + _beta*_n_wrds);
    }

    pi *= temp;
    if (log(urand(_rng)) >= pi){
        // reject
        t_new = t_old;
    }

    _asgmnt[didx][widx] = t_new;
    ++_n_k[t_new];
    ++_n_kw[t_new][wrd];
    ++_n_dk[didx][t_new];

    // XXX: Not for use with sequential importance sampling
    return (pi > 0.0) ? 0.0 : pi;
}


double LightLDA::run(size_t n_steps, bool optimize=false)
{
    vector<Concordance> dw_idx;
    for (size_t didx=0; didx < _n_docs; ++didx){
        size_t widx = 0;
        for (size_t &wrd : _docs[didx]){
            Concordance cnc(didx, wrd, widx);
            dw_idx.push_back(cnc);
            ++widx;
        }
    }

    // Annealing setup.
    // The annealing schdule hs exponential decay and will reach temp=1 once
    // 75% of the steps have been run.
    double T = static_cast<double>(n_steps);
    double alpha = pow(1/T, -4/(3*T));
    double temp = 1;

    double logp = 0;
    for (size_t i=0; i < n_steps; ++i){
        if (optimize){
            double k = static_cast<double>(i);
            temp = static_cast<double>(n_steps)/pow(alpha, k);
        }
        std::shuffle(dw_idx.begin(), dw_idx.end(), _rng);
        for(auto &dw : dw_idx){
            _word_prop = true;
            logp += this->step(dw, temp);
            _word_prop = false;
            logp += this->step(dw, temp);
        }
    }
    return logp;
}


void LightLDA::set_asgn_from_radix(vector<size_t> asgn_radix)
{
    LDA::set_asgn_from_radix(asgn_radix);

    // XXX: This is producing segfault.
    for (auto &table : _alias_tables){
        table.rebuild();
    }

    // TODO: Optimize this.
    // _alias_tables.clear();
    // for (size_t wrd=0; wrd < _n_wrds; ++wrd){
    //     _alias_tables.emplace_back(_n_kw, _n_k, _beta, wrd, _n_tpcs);
    // }
}


// --- SparseLDA
SparseLDA::SparseLDA(vector<vector<size_t>> docs, size_t n_tpcs, size_t n_wrds,
                     double alpha, double beta, int seed) :
    LDA(docs, n_tpcs, n_wrds, alpha, beta, seed)
{
    _cnst.resize(_n_tpcs);
    for (size_t tidx=0; tidx < _n_tpcs; ++tidx)
        _cnst[tidx] = _beta*_n_wrds + _n_k[tidx];

    _s = 0;
    for (auto &c : _cnst)
        _s += 1/c;

    _s *= _alpha*_beta;

    _q_cache.resize(_n_docs, vector<double>(_n_tpcs, 0));
    _doc_tpcs.resize(_n_docs, set<size_t>());
    _wrd_tpcs.resize(_n_wrds, set<size_t>());
    for (size_t didx=0; didx < _n_docs; ++didx){
        for (size_t tidx=0; tidx < _n_tpcs; ++tidx){
            double ct = static_cast<double>(_n_dk[didx][tidx]);
            _q_cache[didx][tidx] = (ct + _alpha)/_cnst[tidx];
        }
        for (size_t widx=0; widx < _docs[didx].size(); ++widx){
            size_t wrd = _docs[didx][widx];
            size_t tpc = _asgmnt[didx][widx];
            _doc_tpcs[didx].insert(tpc);
            _wrd_tpcs[wrd].insert(tpc);
        }
    }
}

void SparseLDA::set_asgn_from_radix(vector<size_t> asgn_radix)
{
    LDA::set_asgn_from_radix(asgn_radix);

    for (size_t tidx=0; tidx < _n_tpcs; ++tidx)
        _cnst[tidx] = _beta*_n_wrds + _n_k[tidx];

    _s = 0;
    for (auto &c : _cnst) _s += 1/c;
    _s *= _alpha*_beta;

    std::fill(_doc_tpcs.begin(), _doc_tpcs.end(), set<size_t>());
    std::fill(_wrd_tpcs.begin(), _wrd_tpcs.end(), set<size_t>());
    for (size_t didx=0; didx < _n_docs; ++didx){
        for (size_t tidx=0; tidx < _n_tpcs; ++tidx){
            double ct = static_cast<double>(_n_dk[didx][tidx]);
            _q_cache[didx][tidx] = (ct + _alpha)/_cnst[tidx];
        }
        for (size_t widx=0; widx < _docs[didx].size(); ++widx){
            size_t wrd = _docs[didx][widx];
            size_t tpc = _asgmnt[didx][widx];
            _doc_tpcs[didx].insert(tpc);
            _wrd_tpcs[wrd].insert(tpc);
        }
    }
}

void SparseLDA::_sub_sr(size_t didx, size_t tidx)
{
    _s -= _alpha*_beta/_cnst[tidx];
    _r -= _beta*_n_dk[didx][tidx]/_cnst[tidx];
}


void SparseLDA::_add_sr(size_t didx, size_t tidx)
{
    _s += _alpha*_beta/_cnst[tidx];
    _r += _beta*_n_dk[didx][tidx]/_cnst[tidx];
}


void SparseLDA::_update_q_cache(size_t didx, size_t tidx)
{
    _q_cache[didx][tidx] = (_alpha + _n_dk[didx][tidx])/_cnst[tidx];
}


void SparseLDA::_remove_word(size_t didx, size_t widx)
{
    const size_t tpc = _asgmnt[didx][widx];
    const size_t wrd = _docs[didx][widx];

    // Mark?
    _sub_sr(didx, tpc);

    --_n_kw[tpc][wrd];
    --_n_dk[didx][tpc];
    --_n_k[tpc];

    _cnst[tpc] = _beta*_n_wrds + _n_k[tpc];

    _add_sr(didx, tpc);
    _update_q_cache(didx, tpc);

    if (_n_kw[tpc][wrd] == 0)
        _wrd_tpcs[wrd].erase(tpc);
    if (_n_dk[didx][tpc] == 0)
        _doc_tpcs[didx].erase(tpc);
}


void SparseLDA::_insert_word(size_t didx, size_t widx, size_t tpc)
{
    const size_t wrd = _docs[didx][widx];

    _asgmnt[didx][widx] = tpc;

    _sub_sr(didx, tpc);

    ++_n_kw[tpc][wrd];
    ++_n_dk[didx][tpc];
    ++_n_k[tpc];

    _cnst[tpc] = _beta*_n_wrds + _n_k[tpc];

    _add_sr(didx, tpc);
    _update_q_cache(didx, tpc);

    if (_n_kw[tpc][wrd] == 1)
        _wrd_tpcs[wrd].insert(tpc);
    if (_n_dk[didx][tpc] == 1)
        _doc_tpcs[didx].insert(tpc);
}


double SparseLDA::step(Concordance dw, double temp)
{
    const auto didx = dw.doc_idx;
    const auto wrd = dw.word;
    const auto widx = dw.word_idx;

    _remove_word(didx, widx);

    double q = 0;
    for (size_t tidx=0; tidx < _n_tpcs; ++tidx)
        q += _q_cache[didx][tidx]*static_cast<double>(_n_kw[tidx][wrd]);

    std::uniform_real_distribution<double> urand(0.0, _s+_r+q);
    double u = urand(_rng);

    size_t t_new;
    double logp = 0;
    if (u < _s){
        vector<double> icnst(_n_tpcs);
        for (size_t tidx=0; tidx < _n_tpcs; ++tidx){
            icnst[tidx] = -log(_cnst[tidx]);
        }
        t_new = log_discrete_draw(icnst, _rng);
        logp = icnst[t_new];
    }else if (u < _s+_r){
        vector<size_t> tpcs(_doc_tpcs[didx].begin(), _doc_tpcs[didx].end());
        vector<double> logps(tpcs.size());
        for(size_t i=0; i < tpcs.size(); ++i){
            size_t tpc = tpcs[i];
            assert(_n_dk[didx][tpc] > 0);
            logps[i] = log(_n_dk[didx][tpc]) + log(_beta/_cnst[tpc]);
        }
        t_new = tpcs[log_discrete_draw(logps, _rng)];
        logp = logps[t_new];
    }else{
        vector<size_t> tpcs(_wrd_tpcs[wrd].begin(), _wrd_tpcs[wrd].end());
        vector<double> logps(tpcs.size());
        for(size_t i=0; i < tpcs.size(); ++i){
            size_t tpc = tpcs[i];
            assert(_n_kw[tpc][wrd] > 0);
            logps[i] = log(_n_kw[tpc][wrd]) + log(_q_cache[didx][tpc]);
        }
        t_new = tpcs[log_discrete_draw(logps, _rng)];
        logp = logps[t_new];
    }

    _insert_word(didx, widx, t_new);

    // not for use with sequential importance sampling
    return logp;
}


double SparseLDA::run(size_t n_steps, bool optimize=false)
{
    // Annealing setup.
    // The annealing schdule hs exponential decay and will reach temp=1 once
    // 75% of the steps have been run.
    double T = static_cast<double>(n_steps);
    double alpha = pow(1/T, -4/(3*T));
    double temp = 1;

    double logp = 0;
    for (size_t i=0; i < n_steps; ++i){
        if (optimize){
            double k = static_cast<double>(i);
            temp = static_cast<double>(n_steps)/pow(alpha, k);
        }
        for (size_t didx=0; didx < _n_docs; ++didx){
            _r = 0;
            for(const auto &tpc : _doc_tpcs[didx]){
                _r += _n_dk[didx][tpc]*_beta/_cnst[tpc];
            }
            size_t widx = 0;
            for (const size_t &wrd : _docs[didx]){
                logp += this->step({didx, wrd, widx}, temp);
                ++widx;
            }
        }
    }
    return logp;
}


// --- LazyLDA
LazyLDA::LazyLDA(vector<vector<size_t>> docs, size_t n_tpcs, size_t n_wrds,
                 double alpha, double beta, int seed) : 
    LDA(docs, n_tpcs, n_wrds, alpha, beta, seed)
{
    _word_prop = true;
    _n = 0;
    for(const auto &doc : docs)
        _n += doc.size();

    _n_w.resize(_n_wrds);
    _z_w.resize(_n_wrds, Counter({}));
    for (size_t didx=0; didx < _n_docs; ++didx){
        const auto &doc = docs[didx];
        for (size_t widx=0; widx < doc.size(); ++widx){
            const auto &wrd = doc[widx];
            ++_n_w[wrd];
            _z_w[wrd].add(_asgmnt[didx][widx]);
        }
    }
}


void LazyLDA::set_asgn_from_radix(vector<size_t> asgn_radix)
{
    std::fill(_n_k.begin(), _n_k.end(), 0);
    std::fill(_z_w.begin(), _z_w.end(), Counter({}));
    for(auto &ndk: _n_dk)
        std::fill(ndk.begin(), ndk.end(), 0);
    for (auto &nkw : _n_kw)
        std::fill(nkw.begin(), nkw.end(), 0);

    size_t ridx = 0;
    for (size_t didx=0; didx < _n_docs; ++didx){
        assert(_docs[didx].size() == _asgmnt[didx].size());
        for (size_t widx=0; widx < _asgmnt[didx].size(); ++widx){
            auto tpc = asgn_radix[ridx++];
            auto wrd = _docs[didx][widx];

            _z_w[wrd].add(tpc);
            _asgmnt[didx][widx] = tpc;

            ++_n_k[tpc];
            ++_n_dk[didx][tpc];
            ++_n_kw[tpc][wrd];
        }
    }

    assert(ridx == asgn_radix.size());
}


double LazyLDA::step(Concordance dw, double temp)
{
    const auto didx = dw.doc_idx;
    const auto wrd = dw.word;
    const auto widx = dw.word_idx;
    const auto t_old = _asgmnt[didx][widx];

    --_n_k[t_old];
    --_n_kw[t_old][wrd];
    --_n_dk[didx][t_old];

    std::uniform_real_distribution<double> urand(0.0, 1.0);

    size_t t_new;
    double pi = 0;
    if(_word_prop){
        double p = _n_w[wrd]-1;
        p /= p + _beta*_n_tpcs*_n_wrds;
        if (urand(_rng) < p){
            t_new = _z_w[wrd].rand_elem(_rng);
        }else{
            std::uniform_int_distribution<size_t> uint(0, _n_tpcs-1);
            t_new = uint(_rng); 
        }
        if (t_old != t_new){
            pi += log(_n_dk[didx][t_new] + _alpha);
            pi -= log(_n_dk[didx][t_old] + _alpha);

            pi += log(_n_kw[t_old][wrd] + _beta*_n_wrds);
            pi -= log(_n_kw[t_new][wrd] + _beta*_n_wrds);
        }
    }else{
        double l = static_cast<double>(_docs[didx].size()-1);
        double p = l / (l + _alpha*_n_tpcs); 
        if (urand(_rng) < p){
            std::uniform_int_distribution<size_t> uint(0, _docs[didx].size()-1);
            size_t tnidx = 0;
            do{
                tnidx = uint(_rng); 
            }while(tnidx == widx);
            t_new = _asgmnt[didx][tnidx];
        }else{
            std::uniform_int_distribution<size_t> uint(0, _n_tpcs-1);
            t_new = uint(_rng); 
        }
    }

    if (t_old != t_new){
        pi += log(_n_kw[t_new][wrd] + _beta);
        pi -= log(_n_kw[t_old][wrd] + _beta);

        pi += log(_n_k[t_old] + _beta*_n_wrds);
        pi -= log(_n_k[t_new] + _beta*_n_wrds);
    }

    pi *= temp;
    if (log(urand(_rng)) < pi){
        _z_w[wrd].subtract(t_old);
        _z_w[wrd].add(t_new);
    }else{
        t_new = t_old;
    }

    _asgmnt[didx][widx] = t_new;
    ++_n_k[t_new];
    ++_n_kw[t_new][wrd];
    ++_n_dk[didx][t_new];

    // XXX: Not for use with sequential importance sampling
    return (pi > 0.0) ? 0.0 : pi;
}

double LazyLDA::run(size_t n_steps, bool optimize=false)
{
    vector<Concordance> dw_idx;
    for (size_t didx=0; didx < _n_docs; ++didx){
        size_t widx = 0;
        for (size_t &wrd : _docs[didx]){
            Concordance cnc(didx, wrd, widx);
            dw_idx.push_back(cnc);
            ++widx;
        }
    }

    // Annealing setup.
    // The annealing schdule hs exponential decay and will reach temp=1 once
    // 75% of the steps have been run.
    double T = static_cast<double>(n_steps);
    double alpha = pow(1/T, -4/(3*T));
    double temp = 1;

    double logp = 0;
    for (size_t i=0; i < n_steps; ++i){
        if (optimize){
            double k = static_cast<double>(i);
            temp = static_cast<double>(n_steps)/pow(alpha, k);
        }
        std::shuffle(dw_idx.begin(), dw_idx.end(), _rng);
        for(auto &dw : dw_idx){
            _word_prop = true;
            logp += this->step(dw, temp);
            _word_prop = false;
            logp += this->step(dw, temp);
        }
    }
    return logp;
}


// -- LDA with known topics
LDAKnownTopics::LDAKnownTopics(
        vector<vector<size_t>> docs, vector<vector<double>> tpcs,
        size_t n_wrds, double alpha, double beta, int seed) : 
    LDA(docs, tpcs.size(), n_wrds, alpha, beta, seed)
{
    _tpcs = tpcs;
}


double LDAKnownTopics::log_conditional(size_t doc, size_t tpc, size_t wrd)
{
    double logp = log(_tpcs[tpc][wrd]) + log(_n_dk[doc][tpc] + _alpha);
    return logp;
}

vector<vector<double>> LDAKnownTopics::get_map_tpcs()
{
    return _tpcs;
}


// -- Standard LDA
LDA::LDA(vector<vector<size_t>> docs, size_t n_tpcs, size_t n_wrds,
         double alpha, double beta, int seed)
{

    _n_docs = docs.size();
    _n_tpcs = n_tpcs;
    _n_wrds = n_wrds;

    _alpha = alpha;
    _beta = beta;

    _n_dk.resize(_n_docs);
    _n_kw.resize(_n_tpcs);
    _n_k.resize(n_tpcs);

    _asgmnt.resize(_n_docs);

    _docs = docs;

    for (auto &row : _n_dk) row.resize(_n_tpcs);
    for (auto &row : _n_kw) row.resize(_n_wrds);

    if ( seed >= 0 ){
        _rng = std::mt19937(seed); 
    } else {
        std::random_device rd;
        _rng = std::mt19937(rd());
    }

    std::uniform_int_distribution<size_t> urnd(0, _n_tpcs-1);

    // initialize from the prior, p(z|alpha)
    size_t didx = 0;
    for (auto &doc : _docs){
        vector<size_t> z_doc(doc.size());
        vector<double> theta = dir_rand(alpha, _n_tpcs, _rng);
        for (auto &f : theta){f = log(f);}
        size_t widx = 0;
        for (auto &wrd : doc){
            size_t tpc = log_discrete_draw(theta, _rng, true);
            z_doc[widx] = tpc;

            ++_n_dk[didx][tpc];
            ++_n_kw[tpc][wrd];
            ++_n_k[tpc];

            ++widx;
        }
        _asgmnt[didx] = z_doc;

        ++didx;
    }
}


// ---
double LDA::run(size_t n_steps, bool optimize=false)
{
    vector<Concordance> dw_idx;
    for (size_t didx=0; didx < _n_docs; ++didx){
        size_t widx = 0;
        for (size_t &wrd : _docs[didx]){
            Concordance cnc(didx, wrd, widx);
            dw_idx.push_back(cnc);
            ++widx;
        }
    }

    // Annealing setup.
    // The annealing schdule hs exponential decay and will reach temp=1 once
    // 75% of the steps have been run.
    double T = static_cast<double>(n_steps);
    double alpha = pow(1/T, -4/(3*T));
    double temp = 1;

    double logp = 0;
    for (size_t i=0; i < n_steps; ++i){
        if (optimize){
            double k = static_cast<double>(i);
            temp = static_cast<double>(n_steps)/pow(alpha, k);
        }
        std::shuffle(dw_idx.begin(), dw_idx.end(), _rng);
        for(auto &dw : dw_idx){
            logp += this->step(dw, temp);
        }
    }
    return logp;
}


double LDA::step(Concordance dw, double temp)
{
    auto didx = dw.doc_idx;
    auto wrd = dw.word;
    auto widx = dw.word_idx;
    auto tpc = _asgmnt[didx][widx];

    -- _n_dk[didx][tpc];
    -- _n_kw[tpc][wrd];
    -- _n_k[tpc];

    vector<double> tpc_logps(_n_tpcs);
    for (size_t k=0; k < _n_tpcs; ++k)
        tpc_logps[k] = this->log_conditional(didx, k, wrd)*temp;
    
    size_t new_tpc = log_discrete_draw(tpc_logps, _rng);

    _asgmnt[didx][widx] = new_tpc;

    double t_prob = tpc_logps[new_tpc] - logsumexp(tpc_logps);

    ++ _n_dk[didx][new_tpc];
    ++ _n_kw[new_tpc][wrd];
    ++ _n_k[new_tpc];

    return t_prob;
}


double LDA::log_conditional(size_t doc, size_t tpc, size_t wrd)
{
    double s_0 = log(static_cast<double>(_n_dk[doc][tpc]) + _alpha);
    double s_1 = log(static_cast<double>(_n_kw[tpc][wrd]) + _beta);
    double s_2 = log(static_cast<double>(_n_k[tpc]) + _beta*_n_wrds);

    return s_0 + s_1 - s_2;
}


// ---
void LDA::set_asgn_from_radix(vector<size_t> asgn_radix)
{
    std::fill(_n_k.begin(), _n_k.end(), 0);
    for(auto &ndk: _n_dk)
        std::fill(ndk.begin(), ndk.end(), 0);
    for (auto &nkw : _n_kw)
        std::fill(nkw.begin(), nkw.end(), 0);

    size_t ridx = 0;
    for (size_t didx=0; didx < _n_docs; ++didx){
        assert(_docs[didx].size() == _asgmnt[didx].size());
        for (size_t widx=0; widx < _asgmnt[didx].size(); ++widx){
            auto tpc = asgn_radix[ridx++];
            auto wrd = _docs[didx][widx];

            _asgmnt[didx][widx] = tpc;

            ++_n_k[tpc];
            ++_n_dk[didx][tpc];
            ++_n_kw[tpc][wrd];
        }
    }

    assert(ridx == asgn_radix.size());
}

// --- MAP parameters
vector<vector<double>> LDA::get_map_tpcs()
{
    vector<vector<double>> map_tpcs(_n_tpcs, vector<double>(_n_wrds, 0)); 
    for (size_t k=0; k < _n_tpcs; ++k){
        double n_k = static_cast<double>(_n_k[k]);
        for (size_t w=0; w < _n_wrds; ++w){
            double n_kw = static_cast<double>(_n_kw[k][w]); 
            map_tpcs[k][w] = (n_kw + _beta)/(n_k + _beta*_n_tpcs);
        }
    }
    return map_tpcs;
}

vector<vector<double>> LDA::get_map_theta()
{
    vector<vector<double>> map_theta(_n_docs, vector<double>(_n_tpcs, 0)); 
    for (size_t didx=0; didx < _n_docs; ++didx){
        double n_d = static_cast<double>(_docs[didx].size());
        for (size_t k=0; k < _n_tpcs; ++k){
            double n_dk = static_cast<double>(_n_dk[didx][k]); 
            map_theta[didx][k] = (n_dk+_alpha) / (n_d+_alpha*_n_tpcs);
        }
    }
    return map_theta;
}


// -- Getters
double LDA::likelihood()
{
    auto asgn = this->get_asgn_radix();

    double log_pzw = _n_tpcs*(lgamma(_n_wrds*_beta)-_n_wrds*lgamma(_beta));
    for (size_t tidx=0; tidx < _n_tpcs; ++tidx){
        log_pzw -= lgamma(static_cast<double>(_n_k[tidx]) + _n_wrds*_beta);
        for (size_t widx=0; widx < _n_wrds; ++widx){
            log_pzw += lgamma(static_cast<double>(_n_kw[tidx][widx]) + _beta);
        }
    }

    return log_pzw;
}

vector<size_t> LDA::get_asgn_radix()
{
    vector<size_t> radix;
    for (auto &asgn : _asgmnt)
        for (auto &tpc : asgn)
            radix.push_back(tpc);

    return radix;
}

vector<vector<size_t>> LDA::get_assignment()
{
    return _asgmnt;
}


vector<size_t> LDA::get_n_k()
{
    return _n_k;
}


vector<vector<size_t>> LDA::get_n_dk()
{
    return _n_dk;
}


vector<vector<size_t>> LDA::get_n_kw()
{
    return _n_kw;
}


// ---
vector<double> LDA::doc_topic_distribution(size_t doc_idx)
{
    return multdirpred(_n_dk[doc_idx], _beta);
}


vector<vector<double>> LDA::doc_topic_distributions()
{
    vector<vector<double>> dists(_n_docs);
    for (size_t didx=0; didx < _n_docs; ++didx)
        dists[didx] = this->doc_topic_distribution(didx);

    return dists;
}

// ---
vector<double> LDA::topic_word_distribution(size_t tpc_idx)
{
    return multdirpred(_n_kw[tpc_idx], _alpha);
}


vector<vector<double>> LDA::topic_word_distributions(){
    vector<vector<double>> dists(_n_tpcs);
    for (size_t tidx=0; tidx < _n_tpcs; ++tidx)
        dists[tidx] = this->topic_word_distribution(tidx);

    return dists;
}

// ---
pair<double, double> __ldateach_sngl_sis(const vector<vector<size_t>> &docs,
                                         const vector<vector<double>> &tpcs,
                                         const vector<size_t> &tpclst,
                                         const vector<size_t> &asgn,
                                         const vector<map<size_t, size_t>> &n_dk,
                                         const map<size_t, map<size_t, size_t>> &n_kw,
                                         const map<size_t, size_t> &n_k,
                                         const set<size_t> &tpcs_ex,
                                         const double alpha, const double beta)
{
    double n_wrds = static_cast<double>(tpcs[0].size());
    double n_docs = static_cast<double>(docs.size());
    double n_tpcs = static_cast<double>(tpcs.size());

    // P(z|alpha)
    double log_pz = n_docs*(lgamma(n_tpcs*alpha)-n_tpcs*lgamma(alpha));
    for (size_t didx=0; didx < n_docs; ++didx){
        log_pz -= lgamma(static_cast<double>(docs[didx].size()) + n_tpcs*alpha);
        for (const auto &cts : n_dk.at(didx)){
            log_pz += lgamma(cts.second + alpha);
        }
        // account for the topics that do not exist in asgn
        log_pz += lgamma(alpha)*(n_tpcs-n_dk.at(didx).size());
    }

    // Generate lookup for tpclst
    vector<bool> include_tpc(n_tpcs, false);
    for (auto &tidx : tpclst) include_tpc[tidx] = true;

    double log_pzw = 0;
    double log_denom = 0;
    double log_numer = 0;

    vector<double> ltpc(n_wrds);

    for (const auto &ct : n_kw){
        size_t tidx = ct.first;
        log_pzw = dirdisc_lpdf_sparse(ct.second, n_wrds, beta);
        if (include_tpc[tidx]){
            // optimize: should be a lot faster if we don't look at words that
            // dont occur
            if (tpcs_ex.find(tidx) != tpcs_ex.end()){
                // the topic appears in the assignment
                for (const auto &kw : n_kw.at(tidx)){
                    size_t wrd = kw.first;
                    size_t ct = kw.second;
                    log_numer += ct*log(tpcs[tidx][wrd]);
                }
            }
        }else{
            log_numer += log_pzw;
        }
        log_denom += log_pzw;
    }
    return {log_numer + log_pz, log_denom + log_pz};
}

// ---
pair<double, double> __ldateach_sngl(const vector<vector<size_t>> &docs,
    const vector<vector<double>> &tpcs, const vector<size_t> &tpclst,
    const vector<size_t> &asgn, const size_t n_wrds, const double alpha,
    const double beta)
{

    const size_t n_tpcs = tpcs.size();

    double lp_z = 0;
    double lp_numer = 0;
    double lp_denom = 0;
    size_t sidx = 0;

    vector<vector<size_t>> tpc_cts(n_tpcs, vector<size_t>(n_wrds, 0));

    size_t zdex = 0;
    for (const auto &doc : docs){
        for (const auto &wrd : doc){
            size_t tpc = asgn[zdex];
            tpc_cts[tpc][wrd]++;
            ++zdex;
        }
    }
    
    for (size_t didx=0; didx < docs.size(); ++didx){
        const size_t doc_size = docs[didx].size();
        const size_t start_idx = sidx;
        const size_t stop_idx = start_idx + doc_size-1;

        const auto tpc_doc_count = vec_to_count(n_tpcs, asgn, start_idx,
                                                stop_idx);
        lp_z += dirdisc_lpdf(tpc_doc_count, alpha);

        assert(sidx < asgn.size());
        sidx = stop_idx + 1;
    }

    assert(sidx == asgn.size());

    // Generate lookup for tpcslst
    vector<bool> include_tpc(n_tpcs, false);
    for (auto &tidx : tpclst) include_tpc[tidx] = true;

    for (size_t tidx=0; tidx < n_tpcs; ++tidx){
        double tpc_lpdf = dirdisc_lpdf(tpc_cts[tidx], beta);
        if (include_tpc[tidx]){
            vector<double> ltpc(n_wrds);
            transform(tpcs[tidx].begin(), tpcs[tidx].end(), ltpc.begin(), log);
            lp_numer += cat_lpdf(tpc_cts[tidx], ltpc);
            if (tidx == n_tpcs-1) assert(n_tpcs == tpcs.size());
        } else {
            assert(tpclst.size() <  tpcs.size());
            assert(tpclst.size() <  n_tpcs);
            lp_numer += tpc_lpdf;
        }
        lp_denom += tpc_lpdf;
    }

    return {lp_numer + lp_z, lp_denom + lp_z};
}


Estimator ldateach_exact(const vector<vector<size_t>> &docs,
                         const vector<vector<double>> &tpcs,
                         const vector<size_t> &tpc_lst,
                         const double alpha, const double beta)
{

    const size_t n_tpcs = tpcs.size();
    const size_t n_wrds = tpcs[0].size();

    size_t asgn_len = 0;
    for (size_t i=0; i < docs.size(); ++i) asgn_len += docs[i].size();

    vector<size_t> asgn(asgn_len);

    const size_t n_asgn = static_cast<size_t>(pow(n_tpcs, asgn_len)+.5);
    vector<double> numer(n_asgn);
    vector<double> denom(n_asgn);

    size_t idx = 0;
    do {
        auto vals = __ldateach_sngl(docs, tpcs, tpc_lst, asgn, n_wrds, alpha,
                                    beta);
        numer[idx] = vals.first;
        denom[idx] = vals.second;

        ++idx;

    } while(radix_incr(asgn, n_tpcs));

    double lp_tpcs = 0;
    for (auto tidx : tpc_lst)
        lp_tpcs += dir_lpdf(tpcs[tidx], beta);

    return {lp_tpcs, numer, denom};
}


Estimator ldateach_unis(const vector<vector<size_t>> &docs,
                        const vector<vector<double>> &tpcs,
                        const vector<size_t> &tpc_lst,
                        const double alpha, const double beta,
                        const size_t n_samples, const unsigned int seed)
{
    std::mt19937 rng(seed);

    const size_t n_wrds = tpcs[0].size();

    size_t asgn_len = 0;
    double n_tpcs = static_cast<double>(tpcs.size());

    for (auto &doc : docs)
        asgn_len += doc.size();

    vector<size_t> asgn(asgn_len);
    vector<double> numer(n_samples);
    vector<double> denom(n_samples);

    std::uniform_int_distribution<size_t> rndtpc(0, n_tpcs-1);

    for (size_t i=0; i < n_samples; ++i){
        // draws z completely at random
        // TODO: Draw z according to p(z | docs, phi) or p(z | docs).
        for (auto &z : asgn) z = rndtpc(rng);

        auto vals = __ldateach_sngl(docs, tpcs, tpc_lst, asgn, n_wrds, alpha,
                                    beta);

        // XXX: q (the value of the importance function) is not included,
        // because it cancels out of the ratio. If you only want part of the
        // estimator, then you'll need to add q back in yourself.
        numer[i] = vals.first;
        denom[i] = vals.second;
    }

    double lp_tpcs = 0;
    for (auto tidx : tpc_lst)
        lp_tpcs += dir_lpdf(tpcs[tidx], beta);

    return {lp_tpcs, numer, denom};
}


Estimator ldateach_pgis(const vector<vector<size_t>> &docs,
                        const vector<vector<double>> &tpcs,
                        const vector<size_t> &tpc_lst,
                        const double alpha, const double beta,
                        const size_t n_samples, const unsigned int seed)
{
    std::mt19937 rng(seed);
   
    const size_t n_docs = docs.size();
    const size_t n_tpcs = tpcs.size();
    // const size_t n_wrds_t = tpcs[0].size();
    const double n_wrds_d = static_cast<double>(tpcs[0].size());

    vector<pair<size_t, size_t>> dw_idx;
    for (size_t didx=0; didx < n_docs; ++didx)
        for (const auto &wrd : docs[didx])
            dw_idx.push_back({didx, wrd});

    size_t asgn_len = dw_idx.size();

    vector<size_t> idxs(asgn_len);
    for (size_t idx=0; idx < asgn_len; ++idx) idxs[idx] = idx;

    vector<double> numer(n_samples);
    vector<double> denom(n_samples);
    
    std::uniform_int_distribution<size_t> utpc(0, n_tpcs-1);

    double logpdef = log(alpha) + log(beta) - log(beta*n_wrds_d);

    for (size_t sample=0; sample < n_samples; ++sample){
        // n_dk[d][k] -> number of words assigned to topic k in document d
        vector<map<size_t, size_t>> n_dk(n_docs);
        // n_kw[k][w] -> number of times word w is assigned to topic k
        map<size_t, map<size_t, size_t>> n_kw;
        // n_k[k] -> total number of words assigned to topic k
        map<size_t, size_t> n_k;
        // list of topics appearing in the assignment
        vector<size_t> tpcs_ex;
        // list of topics not appearing in the assignment
        vector<size_t> tpcs_nex(n_tpcs);

        for (size_t t=0; t < n_tpcs; ++t) tpcs_nex[t] = t;

        vector<size_t> asgn(asgn_len);
        double lq = 0;

        std::shuffle(idxs.begin(), idxs.end(), rng);

        size_t i = 0;
        for (const size_t &idx : idxs){
            size_t tpc = 0;
            double lq_i;
            const auto didx = dw_idx[idx].first;
            const auto wrd = dw_idx[idx].second;
            if (i==0){
                tpc = utpc(rng);

                lq_i = -log(static_cast<double>(n_tpcs));

                tpcs_nex.erase(tpcs_nex.begin()+tpc);
                tpcs_ex.push_back(tpc);
            }else{
                // conditional logP of assignment to each category
                vector<double> logps(tpcs_ex.size()+1);
                size_t t = 0;
                for (const size_t &tidx : tpcs_ex){
                    double ndk = n_dk[didx][tidx];
                    double nkw = n_kw[tidx][wrd];
                    double nk = n_k[tidx];

                    logps[t] = log(ndk + alpha) + log(nkw + beta);
                    logps[t] -= log(nk + beta*n_wrds_d);
                    ++t;
                }
                logps.back() = logpdef +\
                    log(static_cast<double>(tpcs_nex.size()));
                tpc = log_discrete_draw(logps, rng);

                double norm_const = logsumexp(logps);

                lq_i = logps[tpc] - norm_const;

                if (tpc >= tpcs_ex.size()){
                    std::uniform_int_distribution<size_t> urnd(
                        0, tpcs_nex.size()-1);
                    size_t tidx = urnd(rng);
                    lq_i -= log(static_cast<double>(tpcs_nex.size()));
                    tpc = tpcs_nex[tidx];
                    tpcs_nex.erase(tpcs_nex.begin()+tidx);
                    tpcs_ex.push_back(tpc);
                }else{
                    tpc = tpcs_ex[tpc];
                }
            }

            ++n_dk[didx][tpc];
            ++n_kw[tpc][wrd];
            ++n_k[tpc];

            lq += lq_i;

            asgn[idx] = tpc;
            ++i;
        }

        // TODO: you already have the counts, make another calculator that
        // takes counts.
        const set<size_t> tpcs_ex_set(tpcs_ex.begin(), tpcs_ex.end());
        auto vals = __ldateach_sngl_sis(docs, tpcs, tpc_lst, asgn, n_dk, n_kw,
                                        n_k, tpcs_ex_set, alpha, beta);
        numer[sample] = vals.first - lq;
        denom[sample] = vals.second - lq;
    }

    double lp_tpcs = 0;
    for (auto tidx : tpc_lst)
        lp_tpcs += dir_lpdf(tpcs[tidx], beta);

    return {lp_tpcs, numer, denom};
}
