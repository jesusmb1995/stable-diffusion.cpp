// Tests for ggml_backend_sched integration in GGMLRunner.
//
// Validates multi-backend scheduling at the ggml level (unit tests) and
// verifies bitwise-identical output between single-backend and multi-backend
// scheduler paths via PSNR comparison (integration test).

#include <cmath>
#include <cstring>
#include <functional>
#include <numeric>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include "ggml_extend.hpp"

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

// CPU-CPU: same backend type, results must be bitwise identical.
static constexpr double PSNR_BITWISE_DB = 200.0;

// GPU-CPU: cross-device results may differ due to floating-point rounding.
// 55 dB is near-lossless (error is negligible relative to signal).
static constexpr double PSNR_NEAR_LOSSLESS_DB = 55.0;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static double compute_psnr(const float* a, const float* b, size_t n) {
    if (n == 0) return INFINITY;

    double mse = 0.0;
    for (size_t i = 0; i < n; i++) {
        double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        mse += diff * diff;
    }
    mse /= static_cast<double>(n);

    if (mse == 0.0) return INFINITY;

    double max_val = 0.0;
    for (size_t i = 0; i < n; i++) {
        double abs_val = std::fabs(static_cast<double>(a[i]));
        if (abs_val > max_val) max_val = abs_val;
    }
    if (max_val == 0.0) return INFINITY;

    return 10.0 * std::log10((max_val * max_val) / mse);
}

// Minimal concrete GGMLRunner subclass for testing the scheduler path.
// Runs a user-supplied graph-builder function through GGMLRunner::compute().
struct TestRunner : public GGMLRunner {
    std::string desc_;

    TestRunner(ggml_backend_t backend, bool offload = false)
        : GGMLRunner(backend, offload), desc_("TestRunner") {}

    std::string get_desc() override { return desc_; }
};

// ---------------------------------------------------------------------------
// Unit tests: ggml_backend_sched API
// ---------------------------------------------------------------------------

class BackendSchedTest : public ::testing::Test {
protected:
    void SetUp() override {
        cpu1_ = ggml_backend_cpu_init();
        cpu2_ = ggml_backend_cpu_init();
        ASSERT_NE(cpu1_, nullptr);
        ASSERT_NE(cpu2_, nullptr);
    }

    void TearDown() override {
        ggml_backend_free(cpu1_);
        ggml_backend_free(cpu2_);
    }

    ggml_backend_t cpu1_ = nullptr;
    ggml_backend_t cpu2_ = nullptr;
};

TEST_F(BackendSchedTest, TwoBackendMatmulAdd) {
    ggml_backend_t backends[] = {cpu1_, cpu2_};
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        backends, nullptr, 2, 512, false, true);
    ASSERT_NE(sched, nullptr);

    const int M = 2, K = 3;
    ggml_init_params params = {
        ggml_tensor_overhead() * 10 + ggml_graph_overhead(),
        nullptr, true
    };
    ggml_context* ctx = ggml_init(params);
    ASSERT_NE(ctx, nullptr);

    ggml_tensor* a    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    ggml_tensor* b    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, 1);
    ggml_tensor* bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, M);
    ggml_set_name(a, "A"); ggml_set_input(a);
    ggml_set_name(b, "B"); ggml_set_input(b);
    ggml_set_name(bias, "bias"); ggml_set_input(bias);

    ggml_tensor* result = ggml_add(ctx, ggml_mul_mat(ctx, a, b), bias);
    ggml_set_name(result, "result");
    ggml_set_output(result);

    ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);

    ggml_backend_sched_reset(sched);
    ASSERT_TRUE(ggml_backend_sched_alloc_graph(sched, graph));

    float a_data[]    = {1, 2, 3, 4, 5, 6};
    float b_data[]    = {1, 1, 1};
    float bias_data[] = {10, 20};
    ggml_backend_tensor_set(a, a_data, 0, sizeof(a_data));
    ggml_backend_tensor_set(b, b_data, 0, sizeof(b_data));
    ggml_backend_tensor_set(bias, bias_data, 0, sizeof(bias_data));

    ASSERT_EQ(ggml_backend_sched_graph_compute(sched, graph), GGML_STATUS_SUCCESS);

    float output[2];
    ggml_backend_tensor_get(result, output, 0, sizeof(output));

    // A*B = [1+2+3, 4+5+6] = [6, 15];  + bias = [16, 35]
    EXPECT_NEAR(output[0], 16.0f, 1e-5f);
    EXPECT_NEAR(output[1], 35.0f, 1e-5f);

    ggml_free(ctx);
    ggml_backend_sched_free(sched);
}

TEST_F(BackendSchedTest, SingleBackendSquare) {
    ggml_backend_t backends[] = {cpu1_};
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        backends, nullptr, 1, 512, false, true);
    ASSERT_NE(sched, nullptr);

    ggml_init_params params = {
        ggml_tensor_overhead() * 10 + ggml_graph_overhead(),
        nullptr, true
    };
    ggml_context* ctx = ggml_init(params);

    ggml_tensor* x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_set_name(x, "x"); ggml_set_input(x);

    ggml_tensor* result = ggml_sqr(ctx, x);
    ggml_set_name(result, "result");
    ggml_set_output(result);

    ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);

    ggml_backend_sched_reset(sched);
    ggml_backend_sched_alloc_graph(sched, graph);

    float x_data[] = {1, 2, 3, 4};
    ggml_backend_tensor_set(x, x_data, 0, sizeof(x_data));

    ASSERT_EQ(ggml_backend_sched_graph_compute(sched, graph), GGML_STATUS_SUCCESS);

    float output[4];
    ggml_backend_tensor_get(result, output, 0, sizeof(output));
    EXPECT_NEAR(output[0], 1.0f,  1e-5f);
    EXPECT_NEAR(output[1], 4.0f,  1e-5f);
    EXPECT_NEAR(output[2], 9.0f,  1e-5f);
    EXPECT_NEAR(output[3], 16.0f, 1e-5f);

    ggml_free(ctx);
    ggml_backend_sched_free(sched);
}

TEST_F(BackendSchedTest, SchedCreationWithMultipleBackends) {
    ggml_backend_t backends[] = {cpu1_, cpu2_};
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        backends, nullptr, 2, 4096, false, true);
    EXPECT_NE(sched, nullptr);
    if (sched) ggml_backend_sched_free(sched);
}

// ---------------------------------------------------------------------------
// GGMLRunner API tests
// ---------------------------------------------------------------------------

class RunnerSchedTest : public ::testing::Test {
protected:
    void SetUp() override {
        cpu_ = ggml_backend_cpu_init();
        ASSERT_NE(cpu_, nullptr);
    }

    void TearDown() override {
        ggml_backend_free(cpu_);
    }

    ggml_backend_t cpu_ = nullptr;
};

TEST_F(RunnerSchedTest, EnableBackendSchedToggle) {
    TestRunner runner(cpu_);
    EXPECT_FALSE(runner.has_backend_sched());

    ggml_backend_t fallback = ggml_backend_cpu_init();
    ASSERT_NE(fallback, nullptr);

    runner.enable_backend_sched({fallback});
    EXPECT_TRUE(runner.has_backend_sched());

    ggml_backend_free(fallback);
}

TEST_F(RunnerSchedTest, EnableSchedWithEmptyFallbackIsNoop) {
    TestRunner runner(cpu_);
    runner.enable_backend_sched({});
    EXPECT_FALSE(runner.has_backend_sched());
}

TEST_F(RunnerSchedTest, EnableSchedWithSameBackendIsNoop) {
    // If the fallback is the same as runtime_backend, the chain
    // has only 1 unique backend, so no scheduler is created.
    TestRunner runner(cpu_);
    runner.enable_backend_sched({cpu_});
    EXPECT_FALSE(runner.has_backend_sched());
}

// ---------------------------------------------------------------------------
// Integration test: PSNR comparison between single-backend and scheduler paths
// ---------------------------------------------------------------------------

// Builds a non-trivial graph that exercises multiple op types:
//   1. matmul (simulates a dense layer)
//   2. add (bias)
//   3. relu
//   4. scale
//   5. norm (layer normalization)
// This covers the kind of ops found in a real diffusion model pipeline.
static ggml_cgraph* build_dense_norm_graph(ggml_context* ctx,
                                            ggml_tensor* input,
                                            ggml_tensor* weight,
                                            ggml_tensor* bias) {
    ggml_cgraph* graph = ggml_new_graph(ctx);

    // dense = matmul(weight, input) + bias
    ggml_tensor* dense = ggml_add(ctx, ggml_mul_mat(ctx, weight, input), bias);
    ggml_set_name(dense, "dense");

    // activated = relu(dense)
    ggml_tensor* activated = ggml_relu(ctx, dense);
    ggml_set_name(activated, "activated");

    // scaled = activated * 0.5
    ggml_tensor* scaled = ggml_scale(ctx, activated, 0.5f);
    ggml_set_name(scaled, "scaled");

    // out = norm(scaled)
    ggml_tensor* out = ggml_norm(ctx, scaled, 1e-5f);
    ggml_set_name(out, "result");
    ggml_set_output(out);

    ggml_build_forward_expand(graph, out);
    return graph;
}

static std::vector<float> run_graph_single_backend(
    ggml_backend_t backend,
    const float* input_data, int in_dim,
    const float* weight_data, int out_dim,
    const float* bias_data) {

    ggml_init_params params = {
        ggml_tensor_overhead() * 20 + ggml_graph_overhead(),
        nullptr, true
    };
    ggml_context* ctx = ggml_init(params);

    ggml_tensor* input  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_dim);
    ggml_tensor* weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in_dim, out_dim);
    ggml_tensor* bias   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_dim);
    ggml_set_name(input, "input");   ggml_set_input(input);
    ggml_set_name(weight, "weight"); ggml_set_input(weight);
    ggml_set_name(bias, "bias");     ggml_set_input(bias);

    ggml_cgraph* graph = build_dense_norm_graph(ctx, input, weight, bias);

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    EXPECT_TRUE(ggml_gallocr_reserve(allocr, graph));
    EXPECT_TRUE(ggml_gallocr_alloc_graph(allocr, graph));

    ggml_backend_tensor_set(input, input_data, 0, in_dim * sizeof(float));
    ggml_backend_tensor_set(weight, weight_data, 0, in_dim * out_dim * sizeof(float));
    ggml_backend_tensor_set(bias, bias_data, 0, out_dim * sizeof(float));

    if (ggml_backend_is_cpu(backend)) {
        ggml_backend_cpu_set_n_threads(backend, 1);
    }
    EXPECT_EQ(ggml_backend_graph_compute(backend, graph), GGML_STATUS_SUCCESS);

    ggml_tensor* result = ggml_get_tensor(ctx, "result");
    std::vector<float> output(out_dim);
    ggml_backend_tensor_get(result, output.data(), 0, out_dim * sizeof(float));

    ggml_gallocr_free(allocr);
    ggml_free(ctx);
    return output;
}

static std::vector<float> run_graph_sched(
    ggml_backend_t* backends, int n_backends,
    const float* input_data, int in_dim,
    const float* weight_data, int out_dim,
    const float* bias_data) {

    ggml_init_params params = {
        ggml_tensor_overhead() * 20 + ggml_graph_overhead(),
        nullptr, true
    };
    ggml_context* ctx = ggml_init(params);

    ggml_tensor* input  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_dim);
    ggml_tensor* weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in_dim, out_dim);
    ggml_tensor* bias   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_dim);
    ggml_set_name(input, "input");   ggml_set_input(input);
    ggml_set_name(weight, "weight"); ggml_set_input(weight);
    ggml_set_name(bias, "bias");     ggml_set_input(bias);

    ggml_cgraph* graph = build_dense_norm_graph(ctx, input, weight, bias);

    ggml_backend_sched_t sched = ggml_backend_sched_new(
        backends, nullptr, n_backends, 512, false, true);
    EXPECT_NE(sched, nullptr);

    ggml_backend_sched_reset(sched);
    EXPECT_TRUE(ggml_backend_sched_alloc_graph(sched, graph));

    ggml_backend_tensor_set(input, input_data, 0, in_dim * sizeof(float));
    ggml_backend_tensor_set(weight, weight_data, 0, in_dim * out_dim * sizeof(float));
    ggml_backend_tensor_set(bias, bias_data, 0, out_dim * sizeof(float));

    for (int i = 0; i < n_backends; i++) {
        if (ggml_backend_is_cpu(backends[i])) {
            ggml_backend_cpu_set_n_threads(backends[i], 1);
        }
    }

    EXPECT_EQ(ggml_backend_sched_graph_compute(sched, graph), GGML_STATUS_SUCCESS);

    ggml_tensor* result = ggml_get_tensor(ctx, "result");
    std::vector<float> output(out_dim);
    ggml_backend_tensor_get(result, output.data(), 0, out_dim * sizeof(float));

    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    return output;
}

class PSNRIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        cpu1_ = ggml_backend_cpu_init();
        cpu2_ = ggml_backend_cpu_init();
        ASSERT_NE(cpu1_, nullptr);
        ASSERT_NE(cpu2_, nullptr);

        // Deterministic input: a simple sequence scaled to [-1, 1]
        in_dim_  = 64;
        out_dim_ = 32;
        input_data_.resize(in_dim_);
        weight_data_.resize(in_dim_ * out_dim_);
        bias_data_.resize(out_dim_);

        for (int i = 0; i < in_dim_; i++) {
            input_data_[i] = std::sin(static_cast<float>(i) * 0.1f);
        }
        for (int i = 0; i < in_dim_ * out_dim_; i++) {
            weight_data_[i] = std::cos(static_cast<float>(i) * 0.01f) * 0.1f;
        }
        for (int i = 0; i < out_dim_; i++) {
            bias_data_[i] = static_cast<float>(i) * 0.01f - 0.16f;
        }
    }

    void TearDown() override {
        ggml_backend_free(cpu1_);
        ggml_backend_free(cpu2_);
    }

    ggml_backend_t cpu1_ = nullptr;
    ggml_backend_t cpu2_ = nullptr;
    int in_dim_, out_dim_;
    std::vector<float> input_data_, weight_data_, bias_data_;
};

TEST_F(PSNRIntegrationTest, SchedOutputMatchesSingleBackend) {
    auto baseline = run_graph_single_backend(
        cpu1_, input_data_.data(), in_dim_,
        weight_data_.data(), out_dim_, bias_data_.data());

    ggml_backend_t backends[] = {cpu1_, cpu2_};
    auto sched_output = run_graph_sched(
        backends, 2,
        input_data_.data(), in_dim_,
        weight_data_.data(), out_dim_, bias_data_.data());

    ASSERT_EQ(baseline.size(), sched_output.size());

    double psnr = compute_psnr(baseline.data(), sched_output.data(), baseline.size());

    EXPECT_GT(psnr, PSNR_BITWISE_DB)
        << "PSNR too low (" << psnr << " dB). "
           "Scheduler output diverges from single-backend baseline.";
}

TEST_F(PSNRIntegrationTest, RepeatedSchedRunsAreIdentical) {
    ggml_backend_t backends[] = {cpu1_, cpu2_};

    auto run1 = run_graph_sched(
        backends, 2,
        input_data_.data(), in_dim_,
        weight_data_.data(), out_dim_, bias_data_.data());
    auto run2 = run_graph_sched(
        backends, 2,
        input_data_.data(), in_dim_,
        weight_data_.data(), out_dim_, bias_data_.data());

    ASSERT_EQ(run1.size(), run2.size());

    double psnr = compute_psnr(run1.data(), run2.data(), run1.size());
    EXPECT_GT(psnr, PSNR_BITWISE_DB)
        << "Repeated scheduler runs diverged (PSNR=" << psnr << " dB). "
           "Determinism violation.";
}

TEST_F(PSNRIntegrationTest, LargerGraphConverges) {
    // Stress test with a larger layer: 256 -> 128
    int big_in = 256, big_out = 128;
    std::vector<float> big_input(big_in);
    std::vector<float> big_weight(big_in * big_out);
    std::vector<float> big_bias(big_out);

    for (int i = 0; i < big_in; i++) {
        big_input[i] = std::sin(static_cast<float>(i) * 0.05f) * 0.5f;
    }
    for (int i = 0; i < big_in * big_out; i++) {
        big_weight[i] = std::cos(static_cast<float>(i) * 0.003f) * 0.05f;
    }
    for (int i = 0; i < big_out; i++) {
        big_bias[i] = static_cast<float>(i - big_out / 2) * 0.005f;
    }

    auto baseline = run_graph_single_backend(
        cpu1_, big_input.data(), big_in,
        big_weight.data(), big_out, big_bias.data());

    ggml_backend_t backends[] = {cpu1_, cpu2_};
    auto sched_output = run_graph_sched(
        backends, 2,
        big_input.data(), big_in,
        big_weight.data(), big_out, big_bias.data());

    ASSERT_EQ(baseline.size(), sched_output.size());

    double psnr = compute_psnr(baseline.data(), sched_output.data(), baseline.size());
    EXPECT_GT(psnr, PSNR_BITWISE_DB)
        << "Large graph PSNR too low (" << psnr << " dB).";
}

// ---------------------------------------------------------------------------
// Restricted-backend tests: simulate a backend that does NOT support certain
// ops (like a GPU/NPU would), forcing the scheduler to split the graph and
// fall unsupported ops through to the CPU fallback backend.
//
// We do this by creating a real CPU backend and patching its device's
// supports_op vtable entry to reject specific ops. The backend still computes
// everything via CPU under the hood, so we can verify correctness via PSNR.
// ---------------------------------------------------------------------------

// Wraps any backend with a private device that rejects certain ops in
// supports_op(). A separate ggml_backend_device is allocated so patching
// the vtable doesn't affect other backends sharing the same global device.
//
// The device context is NOT replaced — a global registry maps private_device
// pointers to RestrictedBackend instances. This is critical for non-CPU
// backends (e.g. Vulkan) where other vtable functions use dev->context
// for their native state.
struct RestrictedBackend;
static std::unordered_map<ggml_backend_dev_t, RestrictedBackend*> g_restricted_registry;

struct RestrictedBackend {
    ggml_backend_t backend;
    ggml_backend_device* private_device;
    std::set<enum ggml_op> rejected_ops;
    ggml_backend_dev_t original_device;
    bool owns_backend;

    static bool restricted_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor* op) {
        auto it = g_restricted_registry.find(dev);
        if (it == g_restricted_registry.end()) {
            return true;
        }
        auto* self = it->second;
        if (self->rejected_ops.count(op->op)) {
            return false;
        }
        return self->original_device->iface.supports_op(self->original_device, op);
    }

    void restore() {
        g_restricted_registry.erase(private_device);
        backend->device = original_device;
        delete private_device;
        private_device = nullptr;
    }

    void restore_and_free() {
        restore();
        if (owns_backend) {
            ggml_backend_free(backend);
        }
    }
};

// Wrap an existing backend (caller retains ownership of the backend).
static RestrictedBackend* restrict_backend(ggml_backend_t backend,
                                           std::set<enum ggml_op> rejected_ops) {
    auto* rb = new RestrictedBackend();
    rb->backend = backend;
    rb->rejected_ops = std::move(rejected_ops);
    rb->original_device = backend->device;
    rb->owns_backend = false;

    rb->private_device = new ggml_backend_device(*backend->device);
    rb->private_device->iface.supports_op = RestrictedBackend::restricted_supports_op;

    g_restricted_registry[rb->private_device] = rb;
    backend->device = rb->private_device;
    return rb;
}

// Convenience: create a new CPU backend and restrict it (caller owns both).
static RestrictedBackend* create_restricted_cpu(std::set<enum ggml_op> rejected_ops) {
    ggml_backend_t backend = ggml_backend_cpu_init();
    auto* rb = restrict_backend(backend, std::move(rejected_ops));
    rb->owns_backend = true;
    return rb;
}

class RestrictedBackendTest : public ::testing::Test {
protected:
    void SetUp() override {
        cpu_fallback_ = ggml_backend_cpu_init();
        ASSERT_NE(cpu_fallback_, nullptr);

        cpu_baseline_ = ggml_backend_cpu_init();
        ASSERT_NE(cpu_baseline_, nullptr);

        in_dim_  = 64;
        out_dim_ = 32;
        input_data_.resize(in_dim_);
        weight_data_.resize(in_dim_ * out_dim_);
        bias_data_.resize(out_dim_);

        for (int i = 0; i < in_dim_; i++) {
            input_data_[i] = std::sin(static_cast<float>(i) * 0.1f);
        }
        for (int i = 0; i < in_dim_ * out_dim_; i++) {
            weight_data_[i] = std::cos(static_cast<float>(i) * 0.01f) * 0.1f;
        }
        for (int i = 0; i < out_dim_; i++) {
            bias_data_[i] = static_cast<float>(i) * 0.01f - 0.16f;
        }
    }

    void TearDown() override {
        ggml_backend_free(cpu_fallback_);
        ggml_backend_free(cpu_baseline_);
    }

    ggml_backend_t cpu_fallback_ = nullptr;
    ggml_backend_t cpu_baseline_ = nullptr;
    int in_dim_, out_dim_;
    std::vector<float> input_data_, weight_data_, bias_data_;
};

// Primary backend rejects MUL_MAT (like a GPU that hasn't implemented matmul).
// The scheduler must route MUL_MAT to the CPU fallback while keeping ADD,
// RELU, SCALE, NORM on the primary. Output must match single-backend baseline.
TEST_F(RestrictedBackendTest, RejectMulMat_FallbackProducesCorrectResult) {
    auto* restricted = create_restricted_cpu({GGML_OP_MUL_MAT});

    auto baseline = run_graph_single_backend(
        cpu_baseline_, input_data_.data(), in_dim_,
        weight_data_.data(), out_dim_, bias_data_.data());

    ggml_backend_t backends[] = {restricted->backend, cpu_fallback_};
    auto sched_output = run_graph_sched(
        backends, 2,
        input_data_.data(), in_dim_,
        weight_data_.data(), out_dim_, bias_data_.data());

    restricted->restore_and_free();
    delete restricted;

    ASSERT_EQ(baseline.size(), sched_output.size());
    double psnr = compute_psnr(baseline.data(), sched_output.data(), baseline.size());
    EXPECT_GT(psnr, PSNR_BITWISE_DB)
        << "MUL_MAT rejection: PSNR too low (" << psnr << " dB). "
           "Scheduler fallback produced different results.";
}

// Primary backend rejects NORM (layer normalization). This forces a graph
// split mid-pipeline: matmul+add+relu+scale run on primary, norm on fallback.
TEST_F(RestrictedBackendTest, RejectNorm_FallbackProducesCorrectResult) {
    auto* restricted = create_restricted_cpu({GGML_OP_NORM});

    auto baseline = run_graph_single_backend(
        cpu_baseline_, input_data_.data(), in_dim_,
        weight_data_.data(), out_dim_, bias_data_.data());

    ggml_backend_t backends[] = {restricted->backend, cpu_fallback_};
    auto sched_output = run_graph_sched(
        backends, 2,
        input_data_.data(), in_dim_,
        weight_data_.data(), out_dim_, bias_data_.data());

    restricted->restore_and_free();
    delete restricted;

    ASSERT_EQ(baseline.size(), sched_output.size());
    double psnr = compute_psnr(baseline.data(), sched_output.data(), baseline.size());
    EXPECT_GT(psnr, PSNR_BITWISE_DB)
        << "NORM rejection: PSNR too low (" << psnr << " dB).";
}

// Primary backend rejects multiple ops (MUL_MAT + ADD), keeping only
// elementwise ops. This simulates a very limited accelerator (like an
// early NPU that only supports unary ops).
TEST_F(RestrictedBackendTest, RejectMultipleOps_FallbackProducesCorrectResult) {
    auto* restricted = create_restricted_cpu({GGML_OP_MUL_MAT, GGML_OP_ADD});

    auto baseline = run_graph_single_backend(
        cpu_baseline_, input_data_.data(), in_dim_,
        weight_data_.data(), out_dim_, bias_data_.data());

    ggml_backend_t backends[] = {restricted->backend, cpu_fallback_};
    auto sched_output = run_graph_sched(
        backends, 2,
        input_data_.data(), in_dim_,
        weight_data_.data(), out_dim_, bias_data_.data());

    restricted->restore_and_free();
    delete restricted;

    ASSERT_EQ(baseline.size(), sched_output.size());
    double psnr = compute_psnr(baseline.data(), sched_output.data(), baseline.size());
    EXPECT_GT(psnr, PSNR_BITWISE_DB)
        << "MUL_MAT+ADD rejection: PSNR too low (" << psnr << " dB).";
}

// Primary backend rejects ALL ops in the graph. Every node should fall
// through to the CPU fallback. This is the extreme case — the scheduler
// must handle a backend that supports nothing.
TEST_F(RestrictedBackendTest, RejectAllOps_EverythingFallsToCPU) {
    auto* restricted = create_restricted_cpu({
        GGML_OP_MUL_MAT, GGML_OP_ADD, GGML_OP_UNARY,
        GGML_OP_SCALE, GGML_OP_NORM
    });

    auto baseline = run_graph_single_backend(
        cpu_baseline_, input_data_.data(), in_dim_,
        weight_data_.data(), out_dim_, bias_data_.data());

    ggml_backend_t backends[] = {restricted->backend, cpu_fallback_};
    auto sched_output = run_graph_sched(
        backends, 2,
        input_data_.data(), in_dim_,
        weight_data_.data(), out_dim_, bias_data_.data());

    restricted->restore_and_free();
    delete restricted;

    ASSERT_EQ(baseline.size(), sched_output.size());
    double psnr = compute_psnr(baseline.data(), sched_output.data(), baseline.size());
    EXPECT_GT(psnr, PSNR_BITWISE_DB)
        << "All-ops rejection: PSNR too low (" << psnr << " dB).";
}

// ---------------------------------------------------------------------------
// Vulkan backend tests: real GPU primary with CPU fallback.
// Only compiled when SD_USE_VULKAN is defined (i.e. -DSD_VULKAN=ON).
// ---------------------------------------------------------------------------

#ifdef SD_USE_VULKAN
#include "ggml-vulkan.h"

class VulkanBackendTest : public ::testing::Test {
protected:
    void SetUp() override {
        int dev_count = ggml_backend_vk_get_device_count();
        if (dev_count == 0) {
            GTEST_SKIP() << "No Vulkan devices available";
        }

        vk_ = ggml_backend_vk_init(0);
        if (!vk_) {
            GTEST_SKIP() << "Failed to initialize Vulkan backend";
        }

        cpu_ = ggml_backend_cpu_init();
        ASSERT_NE(cpu_, nullptr);

        in_dim_  = 64;
        out_dim_ = 32;
        input_data_.resize(in_dim_);
        weight_data_.resize(in_dim_ * out_dim_);
        bias_data_.resize(out_dim_);

        for (int i = 0; i < in_dim_; i++) {
            input_data_[i] = std::sin(static_cast<float>(i) * 0.1f);
        }
        for (int i = 0; i < in_dim_ * out_dim_; i++) {
            weight_data_[i] = std::cos(static_cast<float>(i) * 0.01f) * 0.1f;
        }
        for (int i = 0; i < out_dim_; i++) {
            bias_data_[i] = static_cast<float>(i) * 0.01f - 0.16f;
        }
    }

    void TearDown() override {
        if (vk_)  ggml_backend_free(vk_);
        if (cpu_) ggml_backend_free(cpu_);
    }

    ggml_backend_t vk_  = nullptr;
    ggml_backend_t cpu_ = nullptr;
    int in_dim_, out_dim_;
    std::vector<float> input_data_, weight_data_, bias_data_;
};

// Vulkan primary + CPU fallback through the scheduler. The scheduler
// dispatches each op to Vulkan if supports_op() returns true, otherwise
// falls back to CPU. Output is compared against a CPU-only baseline.
TEST_F(VulkanBackendTest, VulkanWithCpuFallback_PSNR) {
    auto baseline = run_graph_single_backend(
        cpu_, input_data_.data(), in_dim_,
        weight_data_.data(), out_dim_, bias_data_.data());

    ggml_backend_t backends[] = {vk_, cpu_};
    auto sched_output = run_graph_sched(
        backends, 2,
        input_data_.data(), in_dim_,
        weight_data_.data(), out_dim_, bias_data_.data());

    ASSERT_EQ(baseline.size(), sched_output.size());

    double psnr = compute_psnr(baseline.data(), sched_output.data(), baseline.size());
    EXPECT_GT(psnr, PSNR_NEAR_LOSSLESS_DB)
        << "Vulkan+CPU scheduler vs CPU-only: PSNR too low (" << psnr << " dB). "
           "GPU/CPU numerical divergence exceeds acceptable threshold.";
}

// Vulkan primary + CPU fallback, repeated twice. Both runs must produce
// identical output (determinism of the Vulkan scheduler path).
TEST_F(VulkanBackendTest, VulkanWithCpuFallback_Determinism) {
    ggml_backend_t backends[] = {vk_, cpu_};

    auto run1 = run_graph_sched(
        backends, 2,
        input_data_.data(), in_dim_,
        weight_data_.data(), out_dim_, bias_data_.data());
    auto run2 = run_graph_sched(
        backends, 2,
        input_data_.data(), in_dim_,
        weight_data_.data(), out_dim_, bias_data_.data());

    ASSERT_EQ(run1.size(), run2.size());

    double psnr = compute_psnr(run1.data(), run2.data(), run1.size());
    EXPECT_GT(psnr, PSNR_BITWISE_DB)
        << "Vulkan scheduler runs not deterministic (PSNR=" << psnr << " dB).";
}

// Larger graph (256->128) through Vulkan+CPU scheduler, compared to CPU baseline.
TEST_F(VulkanBackendTest, VulkanWithCpuFallback_LargerGraph) {
    int big_in = 256, big_out = 128;
    std::vector<float> big_input(big_in);
    std::vector<float> big_weight(big_in * big_out);
    std::vector<float> big_bias(big_out);

    for (int i = 0; i < big_in; i++) {
        big_input[i] = std::sin(static_cast<float>(i) * 0.05f) * 0.5f;
    }
    for (int i = 0; i < big_in * big_out; i++) {
        big_weight[i] = std::cos(static_cast<float>(i) * 0.003f) * 0.05f;
    }
    for (int i = 0; i < big_out; i++) {
        big_bias[i] = static_cast<float>(i - big_out / 2) * 0.005f;
    }

    auto baseline = run_graph_single_backend(
        cpu_, big_input.data(), big_in,
        big_weight.data(), big_out, big_bias.data());

    ggml_backend_t backends[] = {vk_, cpu_};
    auto sched_output = run_graph_sched(
        backends, 2,
        big_input.data(), big_in,
        big_weight.data(), big_out, big_bias.data());

    ASSERT_EQ(baseline.size(), sched_output.size());

    double psnr = compute_psnr(baseline.data(), sched_output.data(), baseline.size());
    EXPECT_GT(psnr, PSNR_NEAR_LOSSLESS_DB)
        << "Vulkan large graph: PSNR too low (" << psnr << " dB).";
}

// Vulkan with NORM rejected: matmul+add+relu+scale execute on Vulkan,
// but norm is forced to CPU. This creates a real graph split with
// cross-device data transfer (GPU→CPU) mid-pipeline.
TEST_F(VulkanBackendTest, VulkanRejectNorm_FallbackToCpu) {
    auto baseline = run_graph_single_backend(
        cpu_, input_data_.data(), in_dim_,
        weight_data_.data(), out_dim_, bias_data_.data());

    auto* restricted = restrict_backend(vk_, {GGML_OP_NORM});

    ggml_backend_t backends[] = {vk_, cpu_};
    auto sched_output = run_graph_sched(
        backends, 2,
        input_data_.data(), in_dim_,
        weight_data_.data(), out_dim_, bias_data_.data());

    restricted->restore();
    delete restricted;

    ASSERT_EQ(baseline.size(), sched_output.size());
    double psnr = compute_psnr(baseline.data(), sched_output.data(), baseline.size());
    EXPECT_GT(psnr, PSNR_NEAR_LOSSLESS_DB)
        << "Vulkan (NORM rejected) vs CPU: PSNR too low (" << psnr << " dB). "
           "Graph split with GPU->CPU transfer produced incorrect results.";
}

// Vulkan with MUL_MAT rejected: the heaviest op (matmul) runs on CPU,
// while lighter elementwise ops (add, relu, scale, norm) stay on Vulkan.
// This tests CPU→GPU data transfer after the CPU-computed matmul.
TEST_F(VulkanBackendTest, VulkanRejectMulMat_FallbackToCpu) {
    auto baseline = run_graph_single_backend(
        cpu_, input_data_.data(), in_dim_,
        weight_data_.data(), out_dim_, bias_data_.data());

    auto* restricted = restrict_backend(vk_, {GGML_OP_MUL_MAT});

    ggml_backend_t backends[] = {vk_, cpu_};
    auto sched_output = run_graph_sched(
        backends, 2,
        input_data_.data(), in_dim_,
        weight_data_.data(), out_dim_, bias_data_.data());

    restricted->restore();
    delete restricted;

    ASSERT_EQ(baseline.size(), sched_output.size());
    double psnr = compute_psnr(baseline.data(), sched_output.data(), baseline.size());
    EXPECT_GT(psnr, PSNR_NEAR_LOSSLESS_DB)
        << "Vulkan (MUL_MAT rejected) vs CPU: PSNR too low (" << psnr << " dB). "
           "Graph split with CPU->GPU transfer produced incorrect results.";
}

#endif // SD_USE_VULKAN
