/*******************************************************************************
* Copyright 2017 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <stdio.h>
#include <float.h>
#include <time.h>

#include <stdexcept>
#include <vector>
#include <string>

#ifndef INPUT_H
#error INPUT_H is not defined
#endif
#include INPUT_H

#define FWD_CONVOLUTION   0
#define BWD_F_CONVOLUTION 1
#define BWD_D_CONVOLUTION 2

// Calculates convolution output dimension using the definition from Caffe
static inline int calc_out_dim(
        int input_dim, int filter_dim, int padd, int stride)
{
    return (input_dim - filter_dim + 2 * padd) / stride + 1;
}

// Calculates number of operations.
static double calc_flops(bool skip_padding, const conv_problem& prob)
{
    double flops;
    // Recalculate output dims here to reduce the number of params
    int OW = calc_out_dim(prob.w, prob.fw, prob.padd, prob.stride);
    int OH = calc_out_dim(prob.h, prob.fh, prob.padd, prob.stride);
    if (skip_padding) {
        flops = 0;
        for (int oh = 0; oh < OH; ++oh)
        for (int fh = 0; fh < prob.fh; ++fh) {
            int ih = oh * prob.stride + fh - prob.padd;
            if (!(ih >= 0 && ih < prob.h))
                continue;
            for (int ow = 0; ow < OW; ++ow)
            for (int fw = 0; fw < prob.fw; ++fw) {
                int iw = ow * prob.stride + fw - prob.padd;
                flops += (iw >= 0 && iw < prob.w);
            }
        }
    } else
        flops = 1.0 * prob.fw * prob.fh * OW * OH;
    int groups = std::max(1, prob.groups);
    return 2.0 * flops * prob.ic * prob.oc * prob.minibatch / groups;
}

struct bench_result {
    double min_ms, max_gflops;
    double avg_ms, avg_gflops;
};

// Returns milliseconds since the start of the epoch
static inline double ms_timer()
{
    struct timespec tv;
    clock_gettime(CLOCK_MONOTONIC, &tv);
    return (1000000000ll * tv.tv_sec + tv.tv_nsec) / 1e6;
}

// Benchmarking loop
template <typename Func>
static inline bench_result timeit(int niters, double flops, Func func)
{
    const double max_ms_total = 3E3; // max milliseconds per problem
    func(); // Warmup
    bench_result result = {DBL_MAX, 0, 0, 0};
    int iters_done = 0;
    for (; iters_done < niters; iters_done++) {
        double ms = ms_timer();
        func();
        ms = ms_timer() - ms;
        result.avg_ms += ms;
        result.min_ms = std::min(result.min_ms, ms);
        if (result.avg_ms > max_ms_total)
            break;
    }
    result.avg_ms /= iters_done + 1;
    result.avg_gflops = flops / result.avg_ms * 1E-6;
    result.max_gflops = flops / result.min_ms * 1E-6;
    return result;
}

static inline void rand_fill(float *data, size_t len)
{
    static bool initialized = false;
    if (!initialized) {
        srand48(1);
        initialized = true;
    }
    for (size_t i = 0; i < len; i++)
        data[i] = drand48();
}

#ifdef USE_MKL

#include "mkl_dnn.h"

#define STR1(x) #x
#define STR(x) STR1(x)

#define CHECK(dnnCall) do { \
    dnnError_t e = dnnCall; \
    if (e != E_SUCCESS) { \
        printf("[%s:%d] %s = %d\n", __FILE__, __LINE__, STR(dnnCall), e); \
        throw std::runtime_error(STR(dnnCall)); \
    } \
} while (0)

static bench_result bench_conv(conv_problem prob, int mode, bool skip_padding)
{
    size_t groups = std::max(1, prob.groups);
    size_t inputSize[] = {prob.w, prob.h, prob.ic, prob.minibatch};
    size_t filterSize[] = {prob.fw, prob.fh,
        prob.ic / groups, prob.oc / groups, groups};
    size_t outputSize[] = {
        calc_out_dim(prob.w, prob.fw, prob.padd, prob.stride),
        calc_out_dim(prob.h, prob.fh, prob.padd, prob.stride),
        prob.oc, prob.minibatch};
    size_t biasSize[] = {prob.oc};
    size_t convolutionStride[] = {prob.stride, prob.stride};
    int inputOffset[] = {-prob.padd, -prob.padd};

    dnnPrimitive_t conv = NULL;
    void* resources[dnnResourceNumber] = {0};

    // Init requested convolution primitive
    std::vector<dnnResourceType_t> active_resource_types;
    if (mode == FWD_CONVOLUTION) {
        CHECK(dnnGroupsConvolutionCreateForwardBias_F32(&conv, NULL,
                    dnnAlgorithmConvolutionDirect, groups, 4, inputSize,
                    outputSize, filterSize, convolutionStride, inputOffset,
                    dnnBorderZeros));
        active_resource_types = {dnnResourceSrc,
            dnnResourceDst, dnnResourceFilter, dnnResourceBias};
    } else if (mode == BWD_D_CONVOLUTION) {
        CHECK(dnnGroupsConvolutionCreateBackwardData_F32(&conv, NULL,
                    dnnAlgorithmConvolutionDirect, groups, 4, inputSize,
                    outputSize, filterSize, convolutionStride, inputOffset,
                    dnnBorderZeros));
        active_resource_types = {dnnResourceDiffSrc,
            dnnResourceDiffDst, dnnResourceFilter};
    } else if (mode == BWD_F_CONVOLUTION) {
        CHECK(dnnGroupsConvolutionCreateBackwardFilter_F32(&conv, NULL,
                    dnnAlgorithmConvolutionDirect, groups, 4, inputSize,
                    outputSize, filterSize, convolutionStride, inputOffset,
                    dnnBorderZeros));
        active_resource_types = {dnnResourceSrc,
            dnnResourceDiffDst, dnnResourceDiffFilter};
    } else
        throw std::runtime_error("Invalid benchmarking mode");

    // Init all resources needed by the current convolution
    for (auto type : active_resource_types) {
        dnnLayout_t layout;
        CHECK(dnnLayoutCreateFromPrimitive_F32(&layout, conv, type));
        CHECK(dnnAllocateBuffer_F32(&resources[type], layout));
        size_t len = dnnLayoutGetMemorySize_F32(layout) / sizeof(float);
        rand_fill(static_cast<float *>(resources[type]), len);
        CHECK(dnnLayoutDelete_F32(layout));
    }

    auto result = timeit(prob.iters, calc_flops(skip_padding, prob),
            [&](){CHECK(dnnExecute_F32(conv, resources));});

    // Release resources
    for (int i = 0; i < dnnResourceNumber; i++)
        dnnReleaseBuffer_F32(resources[i]);
    dnnDelete_F32(conv);

    return result;
}
#endif

#ifdef USE_MKLDNN

#define COMPUTE_BWD_BIAS 0

#include "mkldnn.hpp"

using namespace mkldnn;

static bench_result bench_conv(conv_problem prob, int mode, bool skip_padding)
{
    engine eng(engine::kind::cpu, 0);

    int groups = std::max(1, prob.groups);

    memory::desc src_d({prob.minibatch, prob.ic, prob.w, prob.h},
            memory::data_type::f32, memory::format::any);
    memory::desc dst_d({prob.minibatch, prob.oc,
            calc_out_dim(prob.w, prob.fw, prob.padd, prob.stride),
            calc_out_dim(prob.h, prob.fh, prob.padd, prob.stride)},
            memory::data_type::f32, memory::format::any);
    std::vector<int> fsizes
        = {prob.oc / groups, prob.ic / groups, prob.fw, prob.fh};
    if (groups != 1) fsizes.insert(fsizes.begin(), groups);
    memory::desc filter_d(fsizes, memory::data_type::f32, memory::format::any);
    memory::desc bias_d({prob.oc},
            memory::data_type::f32, memory::format::any);
    memory::dims strides = {prob.stride, prob.stride};
    memory::dims padding = {prob.padd, prob.padd};

    std::shared_ptr<primitive> conv;
    std::shared_ptr<memory> src;
    std::shared_ptr<memory> dst;
    std::shared_ptr<memory> filter;
    std::shared_ptr<memory> bias;

    auto fwd_conv_pd = convolution_forward::primitive_desc(
            {prop_kind::forward_training, algorithm::convolution_direct,
            src_d, filter_d, bias_d, dst_d,
            strides, padding, padding, padding_kind::zero}, eng);

    if (mode == FWD_CONVOLUTION) {
        src.reset(new memory(fwd_conv_pd.src_primitive_desc()));
        dst.reset(new memory(fwd_conv_pd.dst_primitive_desc()));
        filter.reset(new memory(fwd_conv_pd.weights_primitive_desc()));
        bias.reset(new memory(fwd_conv_pd.bias_primitive_desc()));
        conv.reset(new convolution_forward(fwd_conv_pd,
                    *src, *filter, *bias, *dst));
    } else if (mode == BWD_D_CONVOLUTION) {
        auto bwd_d_conv_pd = convolution_backward_data::primitive_desc(
                {algorithm::convolution_direct, src_d, filter_d, dst_d,
                strides, padding, padding, padding_kind::zero}, eng,
                fwd_conv_pd);
        src.reset(new memory(bwd_d_conv_pd.diff_src_primitive_desc()));
        dst.reset(new memory(bwd_d_conv_pd.diff_dst_primitive_desc()));
        filter.reset(new memory(bwd_d_conv_pd.weights_primitive_desc()));
        conv.reset(new convolution_backward_data(bwd_d_conv_pd,
                    *dst, *filter, *src));
    } else if (mode == BWD_F_CONVOLUTION) {
        auto bwd_f_conv_pd = convolution_backward_weights::primitive_desc(
                {algorithm::convolution_direct, src_d, filter_d,
#if COMPUTE_BWD_BIAS
                bias_d,
#endif
                dst_d,
                strides, padding, padding, padding_kind::zero}, eng,
                fwd_conv_pd);
        src.reset(new memory(bwd_f_conv_pd.src_primitive_desc()));
        dst.reset(new memory(bwd_f_conv_pd.diff_dst_primitive_desc()));
        filter.reset(new memory(bwd_f_conv_pd.diff_weights_primitive_desc()));
#if COMPUTE_BWD_BIAS
        bias.reset(new memory(bwd_f_conv_pd.diff_bias_primitive_desc()));
        conv.reset(new convolution_backward_weights(bwd_f_conv_pd,
                    *src, *dst, *filter, *bias));
#else
        conv.reset(new convolution_backward_weights(bwd_f_conv_pd,
                    *src, *dst, *filter));
#endif
    } else
        throw std::runtime_error("Invalid benchmarking mode");

    for (const auto &m : {src, dst, filter, bias}) {
        if (!m.get() || !m->get())
            continue;
        float *data = static_cast<float *>(m->get_data_handle());
        size_t len = m->get_primitive_desc().get_size() / sizeof(float);
        rand_fill(data, len);
    }

    stream str(stream::kind::eager);
    str.submit({*conv}).wait();

    return timeit(prob.iters, calc_flops(skip_padding, prob),
            [&](){str.rerun().wait();});
}
#endif

#ifdef USE_LIBXSMM
#ifdef _OPENMP
#include "omp.h"
#endif
#include "libxsmm.h"

#define STR1(x) #x
#define STR(x) STR1(x)

#define CHECK(libxsmm_dnn_call) do { \
    libxsmm_dnn_err_t e = libxsmm_dnn_call; \
    if (e != LIBXSMM_DNN_SUCCESS && e!= LIBXSMM_DNN_WARN_FALLBACK) { \
        printf("[%s:%d] %s = %s\n", __FILE__, __LINE__, \
                STR(libxsmm_dnn_call), libxsmm_dnn_get_error(e)); \
        throw std::runtime_error(STR(libxsmm_dnn_call)); \
    } \
} while (0)

static bench_result bench_conv(conv_problem prob, int mode, bool skip_padding)
{
    libxsmm_dnn_conv_desc conv_desc;
    conv_desc.N = prob.minibatch;
    conv_desc.C = prob.ic;
    conv_desc.H = prob.h;
    conv_desc.W = prob.w;
    conv_desc.K = prob.oc;
    conv_desc.S = prob.fw;
    conv_desc.R = prob.fh;
    conv_desc.u = prob.stride;
    conv_desc.v = prob.stride;
    conv_desc.pad_h = prob.padd;
    conv_desc.pad_w = prob.padd;
#ifdef LIBXSMM_PADD_INPUT
    conv_desc.pad_h_in = prob.padd;
    conv_desc.pad_w_in = prob.padd;
#else
    conv_desc.pad_h_in = 0;
    conv_desc.pad_w_in = 0;
#endif
    conv_desc.pad_h_out = 0;
    conv_desc.pad_w_out = 0;
#ifdef _OPENMP
    conv_desc.threads = omp_get_max_threads();
#else
    conv_desc.threads = 1;
#endif
#ifdef LIBXSMM_FORCE_ALGO_DIRECT
    conv_desc.algo = LIBXSMM_DNN_CONV_ALGO_DIRECT;
#else
    conv_desc.algo = LIBXSMM_DNN_CONV_ALGO_AUTO;
#endif
    conv_desc.buffer_format = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM;
    conv_desc.filter_format = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM;
    conv_desc.fuse_ops = LIBXSMM_DNN_CONV_FUSE_NONE;
    conv_desc.options = LIBXSMM_DNN_CONV_OPTION_NONE;
    conv_desc.options = LIBXSMM_DNN_CONV_OPTION_WU_EXT_FILTER_REDUCE;
    conv_desc.datatype = LIBXSMM_DNN_DATATYPE_F32;

    int IW = conv_desc.W;
    int IWp = IW + conv_desc.pad_w_in;
    int IH = conv_desc.H;
    int IHp = IH + conv_desc.pad_h_in;

    int OW = calc_out_dim(conv_desc.W,
            conv_desc.S, conv_desc.pad_w, conv_desc.u);
    int OWp = OW + conv_desc.pad_w_out;
    int OH = calc_out_dim(conv_desc.H,
            conv_desc.R, conv_desc.pad_h, conv_desc.v);
    int OHp = OH + conv_desc.pad_h_out;

    size_t input_libxsmm_len = IWp * IHp * conv_desc.C * conv_desc.N;
    float *input_libxsmm = (float *)libxsmm_aligned_malloc(
            sizeof(float) * input_libxsmm_len, 2UL * 1024 * 1024);
    rand_fill(input_libxsmm, input_libxsmm_len);

    size_t output_libxsmm_len = OWp * OHp * conv_desc.K * conv_desc.N;
    float *output_libxsmm = (float *)libxsmm_aligned_malloc(
            sizeof(float) * output_libxsmm_len, 2UL * 1024 * 1024);
    rand_fill(output_libxsmm, output_libxsmm_len);

    size_t filter_libxsmm_len
        = conv_desc.S * conv_desc.R * conv_desc.C * conv_desc.K;
    float *filter_libxsmm = (float *)libxsmm_aligned_malloc(
            sizeof(float) * filter_libxsmm_len, 2UL * 1024 * 1024);
    rand_fill(filter_libxsmm, filter_libxsmm_len);

    libxsmm_dnn_err_t status;
    libxsmm_dnn_layer *libxsmm_handle
        = libxsmm_dnn_create_conv_layer(conv_desc, &status);
    CHECK(status);

    libxsmm_dnn_buffer *libxsmm_input
        = libxsmm_dnn_link_buffer(libxsmm_handle, LIBXSMM_DNN_INPUT,
                input_libxsmm, LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_PTR,
                &status);
    CHECK(status);

    libxsmm_dnn_buffer *libxsmm_output
        = libxsmm_dnn_link_buffer(libxsmm_handle, LIBXSMM_DNN_OUTPUT,
                output_libxsmm, LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_PTR,
                &status);
    CHECK(status);

    libxsmm_dnn_filter *libxsmm_filter
        = libxsmm_dnn_link_filter(libxsmm_handle, LIBXSMM_DNN_FILTER,
                filter_libxsmm, LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_PTR,
                &status);
    CHECK(status);

    CHECK(libxsmm_dnn_bind_buffer(libxsmm_handle, libxsmm_input,
                LIBXSMM_DNN_REGULAR_INPUT));
    CHECK(libxsmm_dnn_bind_buffer(libxsmm_handle, libxsmm_input,
                LIBXSMM_DNN_GRADIENT_INPUT));
    CHECK(libxsmm_dnn_bind_buffer(libxsmm_handle, libxsmm_output,
                LIBXSMM_DNN_REGULAR_OUTPUT));
    CHECK(libxsmm_dnn_bind_buffer(libxsmm_handle, libxsmm_output,
                LIBXSMM_DNN_GRADIENT_OUTPUT));
    CHECK(libxsmm_dnn_bind_filter(libxsmm_handle, libxsmm_filter,
                LIBXSMM_DNN_REGULAR_FILTER));
    CHECK(libxsmm_dnn_bind_filter(libxsmm_handle, libxsmm_filter,
                LIBXSMM_DNN_GRADIENT_FILTER));

    size_t scratch_size = libxsmm_dnn_get_scratch_size(
            libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL, &status);
    CHECK(status);
    void *scratch = (void *)libxsmm_aligned_malloc(scratch_size,
            2UL * 1024 * 1024);
    CHECK(libxsmm_dnn_bind_scratch(libxsmm_handle,
                LIBXSMM_DNN_COMPUTE_KIND_ALL, scratch));

    libxsmm_dnn_compute_kind compute_kind;
    if (mode == FWD_CONVOLUTION)
        compute_kind = LIBXSMM_DNN_COMPUTE_KIND_FWD;
    else if (mode == BWD_D_CONVOLUTION)
        compute_kind = LIBXSMM_DNN_COMPUTE_KIND_BWD;
    else if (mode == BWD_F_CONVOLUTION)
        compute_kind = LIBXSMM_DNN_COMPUTE_KIND_UPD;
    else
        throw std::runtime_error("Invalid benchmarking mode");

    auto result = timeit(prob.iters, calc_flops(skip_padding, prob), [&](){
#ifdef _OPENMP
#pragma omp parallel
#endif
                {
#ifdef _OPENMP
                    int tid = omp_get_thread_num();
#else
                    int tid = 0;
#endif
                    CHECK(libxsmm_dnn_execute_st(libxsmm_handle,
                                compute_kind, 0, tid));
                }
            });

    CHECK(libxsmm_dnn_release_scratch(libxsmm_handle,
                LIBXSMM_DNN_COMPUTE_KIND_ALL));
    CHECK(libxsmm_dnn_release_buffer(libxsmm_handle,
                LIBXSMM_DNN_REGULAR_INPUT));
    CHECK(libxsmm_dnn_release_buffer(libxsmm_handle,
                LIBXSMM_DNN_GRADIENT_INPUT));
    CHECK(libxsmm_dnn_release_buffer(libxsmm_handle,
                LIBXSMM_DNN_REGULAR_OUTPUT));
    CHECK(libxsmm_dnn_release_buffer(libxsmm_handle,
                LIBXSMM_DNN_GRADIENT_OUTPUT));
    CHECK(libxsmm_dnn_release_filter(libxsmm_handle,
                LIBXSMM_DNN_REGULAR_FILTER));
    CHECK(libxsmm_dnn_release_filter(libxsmm_handle,
                LIBXSMM_DNN_GRADIENT_FILTER));

    CHECK(libxsmm_dnn_destroy_buffer(libxsmm_input));
    CHECK(libxsmm_dnn_destroy_buffer(libxsmm_output));
    CHECK(libxsmm_dnn_destroy_filter(libxsmm_filter));
    CHECK(libxsmm_dnn_destroy_conv_layer(libxsmm_handle));

    libxsmm_free(scratch);
    libxsmm_free(input_libxsmm);
    libxsmm_free(filter_libxsmm);
    libxsmm_free(output_libxsmm);

    return result;
}
#endif

static void usage()
{
    printf("Usage: <executable> "
            "[<flops w/ padding> = 1 | <flops w/o padding> = 0]\n");
    exit(-1);
}

static bool match_filter_str(const std::string &str, const std::string &filter)
{
    size_t filter_len = filter.length();

    if (filter[filter.length() - 1] == '$') {
        filter_len--;
        if (str.length() != filter_len)
            return false;
    }

    // aka lame startswith
    bool r = true;
    for (int i = 0; i < filter_len; i++)
        if (i >= str.length() || str[i] != filter[i]) {
            r = false;
            break;
        }
    return r;
}

int main(int argc, char **argv)
{
    if (argc > 5)
        usage();

    bool skip_padding = false;
    if (argc > 1) {
        if (argv[1] == std::string("0"))
            skip_padding = true;
        else if (argv[1] == std::string("1"))
            skip_padding = false;
        else
            usage();
    }

    bool csv_output = false;
    if (argc > 2) {
        if (argv[2] == std::string("--csv-output"))
            csv_output = true;
        else if (argv[2] == std::string("--original-output"))
            csv_output = false;
        else
            usage();
    }

    std::vector<int> enabled_modes
        = {FWD_CONVOLUTION, BWD_F_CONVOLUTION, BWD_D_CONVOLUTION};
    if (argc > 3) {
        if (argv[3] == std::string("--fwd"))
            enabled_modes = {FWD_CONVOLUTION};
        else if (argv[3] == std::string("--bwd_d"))
            enabled_modes = {BWD_D_CONVOLUTION};
        else if (argv[3] == std::string("--bwd_f"))
            enabled_modes = {BWD_F_CONVOLUTION};
        else if (argv[3] == std::string("--all"))
            enabled_modes
                = {FWD_CONVOLUTION, BWD_F_CONVOLUTION, BWD_D_CONVOLUTION};
        else
            usage();
    }

    std::string filter_str;
    if (argc > 4)
        filter_str = argv[4];

    const char *conv_mode_strs[] = {"FWD", "BWD_F", "BWD_D"};
    const char *skip_padding_strs[]
        = {"w/ padding in flops", "w/o padding in flops"};

    for (auto m : enabled_modes) {
        if (!csv_output)
            printf(" %s Convolution\n", conv_mode_strs[m]);
        for (const auto& p : conv_problems) {
            if (!match_filter_str(p.name, filter_str))
                continue;

            auto r = bench_conv(p, m, skip_padding);
            if (csv_output)
                printf("%s,%d,\"%s\",%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%e,%e,%e,%e\n",
                        conv_mode_strs[m], skip_padding, p.name,
                        p.minibatch, p.w, p.h, p.ic, p.oc, p.fw, p.fh,
                        p.stride, p.stride, p.padd, p.padd,
                        r.min_ms, r.max_gflops, r.avg_ms, r.avg_gflops);
            else
                printf("W=%d, H=%d, C=%d, N=%d, K=%d, R=%d, S=%d | "
                        "%s %s min(ms) %.2f; max(gflop/s) %.2f;"
                        "avg(ms) %.2f; avg(gflop/s) %.2f;\n",
                        p.w, p.h, p.ic, p.minibatch, p.oc, p.fw, p.fh,
                        conv_mode_strs[m], skip_padding_strs[skip_padding],
                        r.min_ms, r.max_gflops, r.avg_ms, r.avg_gflops);
            fflush(0);
        }
    }

    return 0;
}
