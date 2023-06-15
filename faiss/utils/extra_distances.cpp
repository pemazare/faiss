/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/utils/extra_distances.h>

#include <omp.h>
#include <algorithm>
#include <cmath>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/utils/utils.h>


namespace faiss {

/***************************************************************************
 * Distance functions (other than L2 and IP)
 ***************************************************************************/

namespace {


template <class VD, class ResultHandler>
void pairwise_distances_processor(
        VD vd,
        int64_t nq,
        const float* xq,
        int64_t nb,
        const float* xb,
        int64_t ldq,
        int64_t ldb,
        ResultHandler &rh) {

    constexpr int64_t bs_q = 6;
    constexpr int64_t bs_b = 5;

    {
        float dis[bs_q * bs_b];

        for (int64_t i0 = 0; i0 < nq; i0 += bs_q) {
            int64_t i1 = std::min(i0 + bs_q, nq);
            const float* xqi = xq + i0 * ldq;
            const float* xbj = xb;
            rh.begin_multiple(i0, i1);

            for (int64_t j0 = 0; j0 < nb; j0 += bs_b) {
                // compute one block of size bs_q * bs_b
                // xqi and xbj point at the beginning of the arrays to compare
                int64_t j1 = std::min(j0 + bs_b, nb);
                const float* xqi2 = xqi;
                float *dd = dis;
                for (int64_t i = i0; i < i1; i++) {
                    const float *xbj2 = xbj;
                    for (int64_t j = j0; j < j1; j++) {
                        *dd++ = vd(xqi2, xbj2);
                        xbj2 += ldb;
                    }
                    xqi2 += ldq;
                }
                xbj += ldb * bs_b;
                rh.add_results(j0, j1, dis);
            }
            rh.end_multiple();
        }

    }
}

template<class ResultHandler>
void pairwise_distances_processor(
        MetricType mt,
        float metric_arg,
        int64_t d,
        int64_t nq,
        const float* xq,
        int64_t nb,
        const float* xb,
        int64_t ldq,
        int64_t ldb,
        ResultHandler &rh) {

    switch (mt) {
#define HANDLE_VAR(kw)                                            \
    case METRIC_##kw: {                                           \
        VectorDistance<METRIC_##kw> vd = {(size_t)d, metric_arg}; \
        pairwise_distances_processor(                        \
                vd, nq, xq, nb, xb, ldq, ldb, rh);          \
        break;                                                    \
    }
        HANDLE_VAR(L2);
        HANDLE_VAR(L1);
        HANDLE_VAR(Linf);
        HANDLE_VAR(Canberra);
        HANDLE_VAR(BrayCurtis);
        HANDLE_VAR(JensenShannon);
        HANDLE_VAR(Lp);
        HANDLE_VAR(Jaccard);
#undef HANDLE_VAR
        default:
            FAISS_THROW_MSG("metric type not implemented");
    }
}

template <class VD>
void pairwise_extra_distances_template(
        VD vd,
        int64_t nq,
        const float* xq,
        int64_t nb,
        const float* xb,
        float* dis,
        int64_t ldq,
        int64_t ldb,
        int64_t ldd) {
#pragma omp parallel for if (nq > 10)
    for (int64_t i = 0; i < nq; i++) {
        const float* xqi = xq + i * ldq;
        const float* xbj = xb;
        float* disi = dis + ldd * i;

        for (int64_t j = 0; j < nb; j++) {
            disi[j] = vd(xqi, xbj);
            xbj += ldb;
        }
    }
}

template <class VD, class C>
void knn_extra_metrics_template(
        VD vd,
        const float* x,
        const float* y,
        size_t nx,
        size_t ny,
        HeapArray<C>* res) {

    HeapResultHandler<C> rh(nx, res->val, res->ids, res->k);
    pairwise_distances_processor(vd, nx, x, ny, y, vd.d, vd.d, rh);

}

template <class VD>
struct ExtraDistanceComputer : FlatCodesDistanceComputer {
    VD vd;
    idx_t nb;
    const float* q;
    const float* b;

    float symmetric_dis(idx_t i, idx_t j) final {
        return vd(b + j * vd.d, b + i * vd.d);
    }

    float distance_to_code(const uint8_t* code) final {
        return vd(q, (float*)code);
    }

    ExtraDistanceComputer(
            const VD& vd,
            const float* xb,
            size_t nb,
            const float* q = nullptr)
            : FlatCodesDistanceComputer((uint8_t*)xb, vd.d * sizeof(float)),
              vd(vd),
              nb(nb),
              q(q),
              b(xb) {}

    void set_query(const float* x) override {
        q = x;
    }
};

} // anonymous namespace


void knn_extra_metrics(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        MetricType mt,
        float metric_arg,
        float *dis,
        idx_t *ids,
        int k) {

    if (is_similarity_metric(mt)) {
        HeapResultHandler<CMin<float, idx_t>> rh(nx, dis, ids, k);
        pairwise_distances_processor(mt, metric_arg, d, nx, x, ny, y, d, d, rh);
    } else  {
        HeapResultHandler<CMax<float, idx_t>> rh(nx, dis, ids, k);
        pairwise_distances_processor(mt, metric_arg, d, nx, x, ny, y, d, d, rh);
    }
}

void pairwise_extra_distances(
        int64_t d,
        int64_t nq,
        const float* xq,
        int64_t nb,
        const float* xb,
        MetricType mt,
        float metric_arg,
        float* dis,
        int64_t ldq,
        int64_t ldb,
        int64_t ldd) {
    if (nq == 0 || nb == 0)
        return;
    if (ldq == -1)
        ldq = d;
    if (ldb == -1)
        ldb = d;
    if (ldd == -1)
        ldd = nb;

    switch (mt) {
#define HANDLE_VAR(kw)                                            \
    case METRIC_##kw: {                                           \
        VectorDistance<METRIC_##kw> vd = {(size_t)d, metric_arg}; \
        pairwise_extra_distances_template(                        \
                vd, nq, xq, nb, xb, dis, ldq, ldb, ldd);          \
        break;                                                    \
    }
        HANDLE_VAR(L2);
        HANDLE_VAR(L1);
        HANDLE_VAR(Linf);
        HANDLE_VAR(Canberra);
        HANDLE_VAR(BrayCurtis);
        HANDLE_VAR(JensenShannon);
        HANDLE_VAR(Lp);
        HANDLE_VAR(Jaccard);
#undef HANDLE_VAR
        default:
            FAISS_THROW_MSG("metric type not implemented");
    }
}



FlatCodesDistanceComputer* get_extra_distance_computer(
        size_t d,
        MetricType mt,
        float metric_arg,
        size_t nb,
        const float* xb) {
    switch (mt) {
#define HANDLE_VAR(kw)                                                 \
    case METRIC_##kw: {                                                \
        VectorDistance<METRIC_##kw> vd = {(size_t)d, metric_arg};      \
        return new ExtraDistanceComputer<VectorDistance<METRIC_##kw>>( \
                vd, xb, nb);                                           \
    }
        HANDLE_VAR(L2);
        HANDLE_VAR(L1);
        HANDLE_VAR(Linf);
        HANDLE_VAR(Canberra);
        HANDLE_VAR(BrayCurtis);
        HANDLE_VAR(JensenShannon);
        HANDLE_VAR(Lp);
        HANDLE_VAR(Jaccard);
#undef HANDLE_VAR
        default:
            FAISS_THROW_MSG("metric type not implemented");
    }
}

} // namespace faiss
