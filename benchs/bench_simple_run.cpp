#include <cstdio>

#include <omp.h>


#include <faiss/MetricType.h>
#include <faiss/utils/random.h>
#include <faiss/utils/utils.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/extra_distances.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexBinaryFlat.h>
#include <faiss/IndexBinaryIVF.h>
#include <faiss/index_factory.h>
#include <faiss/impl/ResultHandler.h>


void bench_float() {
    int nb = 100 * 1000;
    int nq = 10 * 1000;
    int d = 64;
    int k = 10;

    std::vector<float> xall((nq + nb) * d);
    faiss::rand_smooth_vectors(nq + nb, d, xall.data(), 1324);

    const float *xb = xall.data();
    const float *xq = xall.data() + d * nb;

    std::vector<float> D(nq * k);
    std::vector<faiss::idx_t> I(nq * k);

    // omp_set_num_threads(32);

    {
        double t0 = faiss::getmillisecs();
        // brute force search
        faiss::knn_L2sqr(
            xq, xb, d, nq, nb, k,
            D.data(), I.data());
        double t1 = faiss::getmillisecs();
        printf("Brute force search, BLAS: %.3f ms\n", t1 - t0);
    }

    int prev_threshold = faiss::distance_compute_blas_threshold;
    faiss::distance_compute_blas_threshold = nq + 1;
    {
        double t0 = faiss::getmillisecs();
        // brute force search
        faiss::knn_L2sqr(
            xq, xb, d, nq, nb, k,
            D.data(), I.data());
        double t1 = faiss::getmillisecs();
        printf("Brute force search, simple implementation: %.3f ms\n", t1 - t0);
    }
    faiss::distance_compute_blas_threshold = prev_threshold;
    {
        double t0 = faiss::getmillisecs();
        faiss::knn_extra_metrics(
            xq, xb, d, nq, nb, faiss::METRIC_L2, 0, D.data(), I.data(), k);
        double t1 = faiss::getmillisecs();
        printf("Brute force search, extra_distances implementation: %.3f ms\n", t1 - t0);
    }



    std::vector<float> D2(nq * k);
    std::vector<faiss::idx_t> I2(nq * k);

    {
        faiss::IndexHNSWFlat index(d, 32);
        index.add(nb, xb);

        double t0 = faiss::getmillisecs();
        index.search(nq, xq, k, D2.data(), I2.data());
        double t1 = faiss::getmillisecs();
        int nok = 0;
        for(int i = 0; i < nq; i++) {
            nok += faiss::ranklist_intersection_size(
                k, &I[i * k],
                k, &I2[i * k]
            );
        }
        printf("HNSW search: %.3f ms accuracy: %.3f\n",
            t1 - t0, nok / float(nq * k));
    }

}

void bench_ivf_variant(const char *factory_key) {
    int nb = 1000 * 1000;
    int nq = 10 * 1000;
    int d = 64;
    int k = 10;

    std::vector<float> xall((nq + nb) * d);
    faiss::rand_smooth_vectors(nq + nb, d, xall.data(), 1324);

    const float *xb = xall.data();
    const float *xq = xall.data() + d * nb;

    // will return an IndeIVFFlat or an IndexIVFScalarQuantizer depending
    // on the string
    std::unique_ptr<faiss::IndexIVF> index(
        dynamic_cast<faiss::IndexIVF*>(faiss::index_factory(d, factory_key)));

    {
        double t0 = faiss::getmillisecs();
        index->train(nb, xb);
        printf("train %s: %.3f ms\n", factory_key, faiss::getmillisecs() - t0);
        t0 = faiss::getmillisecs();
        index->add(nb, xb);
        printf("add: %.3f ms\n", faiss::getmillisecs() - t0);
    }

    std::vector<float> D(nq * k);
    std::vector<faiss::idx_t> I(nq * k);

    // brute force reference search
    faiss::knn_L2sqr(
        xq, xb, d, nq, nb, k,
        D.data(), I.data());

    std::vector<float> D2(nq * k);
    std::vector<faiss::idx_t> I2(nq * k);

    for (int nprobe: {1, 4, 16, 64}) {
        double t0 = faiss::getmillisecs();
        index->nprobe = nprobe;
        index->search(nq, xq, k, D2.data(), I2.data());
        double t1 = faiss::getmillisecs();
        int nok = 0;
        for(int i = 0; i < nq; i++) {
            nok += faiss::ranklist_intersection_size(
                k, &I[i * k],
                k, &I2[i * k]
            );
        }
        printf("%s nprobe=%d search: %.3f ms accuracy: %.3f\n",
            factory_key, nprobe, t1 - t0, nok / float(nq * k));
    }

}

void bench_binary() {
    int nb = 100 * 1000;
    int nq = 10 * 1000;
    int d = 256;
    int k = 10;

    std::vector<float> xall_f((nq + nb) * d);
    faiss::rand_smooth_vectors(nq + nb, d, xall_f.data(), 1324);

    std::vector<uint8_t> xall(xall_f.size() / 8);
    faiss::real_to_binary(xall_f.size(), xall_f.data(), xall.data());

    int stride = d / 8;
    const uint8_t *xb = xall.data();
    const uint8_t *xq = xall.data() + stride * nb;

    std::vector<int32_t> D(nq * k);
    std::vector<faiss::idx_t> I(nq * k);

    {
        faiss::IndexBinaryFlat index(d);
        index.add(nb, xb);
        double t0 = faiss::getmillisecs();
        index.search(nq, xq, k, D.data(), I.data());
        double t1 = faiss::getmillisecs();
        printf("Binary brute force search: %.3f ms\n", t1 - t0);
    }
    {
        std::vector<int32_t> D2(nq * k);
        std::vector<faiss::idx_t> I2(nq * k);

        faiss::IndexBinaryFlat quantizer(d);
        faiss::IndexBinaryIVF index(
            &quantizer, d, 256
        );
        index.train(nb, xb);
        index.add(nb, xb);
        index.nprobe = 10;
        double t0 = faiss::getmillisecs();
        index.search(nq, xq, k, D2.data(), I2.data());
        double t1 = faiss::getmillisecs();
        int nok = 0;
        for(int i = 0; i < nq; i++) {
            nok += faiss::ranklist_intersection_size(
                k, &I[i * k],
                k, &I2[i * k]
            );
        }
        printf("Binary IVF search: %.3f ms accuracy: %.3f\n",
            t1 - t0, nok / float(nq * k));
    }
}


int main() {

    bench_float();

    bench_binary();

    bench_ivf_variant("IVF4096,Flat");

    bench_ivf_variant("IVF4096,SQfp16");

    bench_ivf_variant("IVF4096,SQ8");

    return 0;
}