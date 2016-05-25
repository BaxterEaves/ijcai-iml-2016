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


#ifndef ldateach_lda_GUARD__
#define ldateach_lda_GUARD__

#define LDATEACH_EXACT 0
#define LDATEACH_UNIS 1
#define LDATEACH_PGIS 2

#include <iostream>
#include <vector>
#include <tuple>
#include <random>
#include <algorithm>
#include <cassert>
#include <climits>
#include <stdexcept>
#include <cmath>
#include <set>
#include <map>

#include "utils.hpp"
#include "dist.hpp"


struct Concordance{
    size_t doc_idx;
    size_t word;
    size_t word_idx;
    Concordance(size_t didx, size_t wrd, size_t widx) : doc_idx(didx),
        word(wrd), word_idx(widx){}
};

struct Estimator{
    double prior;
    std::vector<double> numer;
    std::vector<double> denom;
};


// -- TEACHING
// Calculates the numerator and denominator for a single assignment. If only
// teaching specific topics, specify those topics in tpc_lst.
std::pair<double, double> __ldateach_sngl(
        const std::vector<std::vector<size_t>> &docs,
        const std::vector<std::vector<double>> &tpcs,
        const std::vector<size_t> &tpc_lst,
        const std::vector<size_t> &asgn, size_t n_wrds,
        double alpha, double beta);

std::pair<double, double> __ldateach_sngl_sis(
    const std::vector<std::vector<size_t>> &docs,
    const std::vector<std::vector<double>> &tpcs,
    const std::vector<size_t> &tpclst,
    const std::vector<size_t> &asgn,
    const std::vector<std::map<size_t, size_t>> &n_dk,
    const std::map<size_t, std::map<size_t, size_t>> &n_kw,
    const std::map<size_t, size_t> &n_k,
    const std::set<size_t> &tpcs_ex,
    const double alpha, const double beta);

// returns the exact posterior of the topics given the documents and alpha and
// beta.
Estimator ldateach_exact(const std::vector<std::vector<size_t>> &docs,
                         const std::vector<std::vector<double>> &tpcs,
                         const std::vector<size_t> &tpc_lst,
                         const double alpha, const double beta);

// uniform importance sampling
Estimator ldateach_unis(const std::vector<std::vector<size_t>> &docs,
                        const std::vector<std::vector<double>> &tpcs,
                        const std::vector<size_t> &tpc_lst,
                        const double alpha, const double beta,
                        const size_t n_samples, const unsigned int seed);

// partial gibbs importance sampling
Estimator ldateach_pgis(const std::vector<std::vector<size_t>> &docs,
                        const std::vector<std::vector<double>> &tpcs,
                        const std::vector<size_t> &tpc_lst,
                        const double alpha, const double beta,
                        const size_t n_samples, const unsigned int seed);


// --- LEARNING
class LDA{
    // Latent Dirichlet Allocation (Gibbs sampler)

    public:
        LDA(std::vector<std::vector<size_t>> docs, size_t n_tpcs,
            size_t n_wrds, double alpha, double beta, int seed);

        // Run the sampler for n_itertions over all words in all documents
        double run(size_t n_steps, bool optimize);
        // One sweep of the sampler over a specific set of documents and words.
        double step(Concordance dw, double temp);
        
        // Set the assignment by  a 1-D vector assignment. radix_asgn is
        // is assumed to be traversed in the same order as _asgn; that is
        // radix_asgn[0] should be _asgn[0][0], radix_asgn[1] should be
        // _asgn[0][1], and radix_asign.back() should be _asgn.back().back().
        void set_asgn_from_radix(std::vector<size_t> radix_asgn);
        std::vector<size_t> get_asgn_radix();

        virtual double log_conditional(size_t doc, size_t tpc, size_t wrd);

        virtual std::vector<std::vector<double>> get_map_tpcs();
        std::vector<std::vector<double>> get_map_theta();
        std::vector<std::vector<size_t>> get_assignment();
        std::vector<std::vector<size_t>> get_n_dk();
        std::vector<std::vector<size_t>> get_n_kw();
        std::vector<size_t> get_n_k();
        double likelihood();

        // predictive distribution of words to topics
        std::vector<double> topic_word_distribution(size_t tpc_idx);
        std::vector<std::vector<double>> topic_word_distributions();

        // predictive distribution of topics to documents 
        std::vector<double> doc_topic_distribution(size_t doc_idx);
        std::vector<std::vector<double>> doc_topic_distributions();

    protected:
        size_t _n_tpcs;
        size_t _n_docs;
        size_t _n_wrds;

        // [d][k] -> number of words assigned to topic k in document d
        std::vector<std::vector<size_t>> _n_dk;
        // [k][w] -> number of times word w is assigned to topic k
        std::vector<std::vector<size_t>> _n_kw;
        // [k] -> total number of words assigned to topic k
        std::vector<size_t> _n_k;

        // [d][w] the id of the wth word in document d
        std::vector<std::vector<size_t>> _docs;

        // [d][w] the topic of the wth word in document d
        std::vector<std::vector<size_t>> _asgmnt;

        // Dirichlet concentation parameter for word/topic distribution 
        double _alpha;
        // Dirichlet concentation parameter for topic/document distribution 
        double _beta;

        std::mt19937 _rng;
};


class LDAKnownTopics : public LDA{
    // LDA assuming that the topics are known
    public:
        LDAKnownTopics(std::vector<std::vector<size_t>> docs,
                       std::vector<std::vector<double>> tpcs,
                       size_t n_wrds, double alpha, double beta, int seed);

        double log_conditional(size_t doc, size_t tpc, size_t wrd);
        std::vector<std::vector<double>> get_map_tpcs();

    protected:
        std::vector<std::vector<double>> _tpcs;
};


class LazyLDA : public LDA{
    public:
        LazyLDA(std::vector<std::vector<size_t>> docs,
                size_t n_tpcs, size_t n_wrds, double alpha, double beta,
                int seed);

        double run(size_t n_steps, bool optimize);
        double step(Concordance dw, double temp);
        void set_asgn_from_radix(std::vector<size_t> radix_asgn);

    private:
        bool _word_prop;
        size_t _n;
        std::vector<size_t> _n_w;
        std::vector<Counter> _z_w;
};


class LightLDA : public LDA{
    public:
        LightLDA(std::vector<std::vector<size_t>> docs,
                size_t n_tpcs, size_t n_wrds, double alpha, double beta,
                int seed);

        double run(size_t n_steps, bool optimize);
        double step(Concordance dw, double temp);
        void set_asgn_from_radix(std::vector<size_t> radix_asgn);

    private:
        bool _word_prop;
        std::vector<AliasTable_LDA> _alias_tables;
};


class SparseLDA : public LDA{
    public:
        SparseLDA(std::vector<std::vector<size_t>> docs,
                  size_t n_tpcs, size_t n_wrds, double alpha, double beta,
                  int seed);

        double run(size_t n_steps, bool optimize);
        double step(Concordance dw, double temp);
        void set_asgn_from_radix(std::vector<size_t> radix_asgn);

    private:
        // all the bookkeeping functions
        void _add_sr(size_t didx, size_t tpc);
        void _sub_sr(size_t didx, size_t tpc);
        void _update_q_cache(size_t didx, size_t tpc);
        void _remove_word(size_t didx, size_t widx);
        void _insert_word(size_t didx, size_t widx, size_t tpc);

        std::vector<double> _cnst;
        std::vector<std::vector<double>> _q_cache;

        double _s;
        double _r;

        std::vector<std::set<size_t>> _doc_tpcs;
        std::vector<std::set<size_t>> _wrd_tpcs;
};

#endif
