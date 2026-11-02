// Integration tests against an unmodified ggml build, using its CPU backend as the oracle.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#x); std::exit(1); } } while (0)

static std::vector<float> read(ggml_tensor * t) {
    std::vector<float> data(ggml_nelements(t));
    ggml_backend_tensor_get(t,data.data(),0,data.size()*sizeof(float));
    return data;
}
static void compare(ggml_tensor * t, const std::vector<float> & want) {
    const auto got = read(t);
    CHECK(got.size()==want.size());
    for (size_t i=0;i<got.size();++i) {
        if (!std::isfinite(got[i]) || std::abs(got[i]-want[i]) > 2e-5f + 2e-4f*std::abs(want[i])) {
            std::fprintf(stderr,"mismatch %s[%zu]: %.9g != %.9g\n",ggml_get_name(t),i,got[i],want[i]);
            std::exit(1);
        }
    }
}
static void fill(ggml_tensor * t, int seed) {
    std::vector<float> data(ggml_nelements(t));
    for (size_t i=0;i<data.size();++i) data[i] = (float(int((i*7+seed*3)%23)-11))/13.0f;
    ggml_backend_tensor_set(t,data.data(),0,data.size()*sizeof(float));
}

static void graph_case(ggml_backend_t bridge, ggml_backend_t cpu, int k, int n, int m, bool view, bool own_buffer) {
    auto * ctx = ggml_init({4*1024*1024,nullptr,true}); CHECK(ctx);
    auto * weights = ggml_new_tensor_2d(ctx,GGML_TYPE_F32,k,n);
    auto * backing = ggml_new_tensor_2d(ctx,GGML_TYPE_F32,view ? k+2 : k,m);
    auto * x = view ? ggml_view_2d(ctx,backing,k,m,backing->nb[1],sizeof(float)) : backing;
    auto * bias = ggml_new_tensor_1d(ctx,GGML_TYPE_F32,n);
    auto * mm = ggml_mul_mat(ctx,weights,x); ggml_set_name(mm,"matmul");
    auto * add = ggml_add(ctx,mm,bias); ggml_set_name(add,"bias");
    auto * silu = ggml_silu(ctx,add); ggml_set_name(silu,"silu");
    auto * mul = ggml_mul(ctx,silu,bias); ggml_set_name(mul,"mul");
    auto * y = ggml_tanh(ctx,mul); ggml_set_name(y,"tanh");
    auto * g = ggml_new_graph(ctx); ggml_build_forward_expand(g,y);
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx,own_buffer ? bridge : cpu); CHECK(buffer);
    for (auto * t : {mm,add,silu,mul,y}) CHECK(ggml_backend_dev_supports_op(ggml_backend_get_device(bridge),t));
    for (int run=0;run<3;++run) {
        fill(weights,run+1); fill(backing,run+3); fill(bias,run+5);
        CHECK(ggml_backend_graph_compute(cpu,g)==GGML_STATUS_SUCCESS);
        std::vector<std::vector<float>> expected;
        for (auto * t : {mm,add,silu,mul,y}) expected.push_back(read(t));
        for (auto * t : {mm,add,silu,mul,y}) ggml_backend_tensor_memset(t,0xff,0,ggml_nbytes(t));
        CHECK(ggml_backend_graph_compute(bridge,g)==GGML_STATUS_SUCCESS);
        int index=0;
        for (auto * t : {mm,add,silu,mul,y}) compare(t,expected[index++]);
    }
    ggml_backend_buffer_free(buffer); ggml_free(ctx);
}

static void transpose_case(ggml_backend_t bridge, ggml_backend_t cpu) {
    auto * ctx = ggml_init({4*1024*1024,nullptr,true}); CHECK(ctx);
    auto * w = ggml_new_tensor_2d(ctx,GGML_TYPE_F32,7,5);
    auto * x = ggml_new_tensor_2d(ctx,GGML_TYPE_F32,3,7);
    auto * xt = ggml_transpose(ctx,x);
    auto * y = ggml_mul_mat(ctx,w,xt);
    auto * reference = ggml_mul_mat(ctx,w,ggml_cont(ctx,xt));
    auto * g = ggml_new_graph(ctx); ggml_build_forward_expand(g,y);
    auto * ref_graph = ggml_new_graph(ctx); ggml_build_forward_expand(ref_graph,reference);
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx,cpu); CHECK(buffer);
    fill(w,2); fill(x,5);
    CHECK(ggml_backend_graph_compute(cpu,ref_graph)==GGML_STATUS_SUCCESS);
    auto expected=read(reference);
    CHECK(ggml_backend_graph_compute(bridge,g)==GGML_STATUS_SUCCESS);
    compare(y,expected);
    ggml_backend_buffer_free(buffer); ggml_free(ctx);
}

static void fallback_case(ggml_backend_t bridge, ggml_backend_t cpu) {
    auto * ctx = ggml_init({4*1024*1024,nullptr,true}); CHECK(ctx);
    auto * w = ggml_new_tensor_2d(ctx,GGML_TYPE_F32,7,5);
    auto * x = ggml_new_tensor_2d(ctx,GGML_TYPE_F32,7,3);
    auto * mm = ggml_mul_mat(ctx,w,x);
    auto * y = ggml_relu(ctx,mm); // deliberately unsupported by the bridge
    ggml_set_output(mm); ggml_set_output(y);
    auto * g = ggml_new_graph(ctx); ggml_build_forward_expand(g,y);
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx,cpu); CHECK(buffer);
    fill(w,1); fill(x,2);
    CHECK(ggml_backend_graph_compute(cpu,g)==GGML_STATUS_SUCCESS);
    auto expected=read(y);
    CHECK(!ggml_backend_dev_supports_op(ggml_backend_get_device(bridge),y));
    // Failure preflight must not execute the supported prefix.
    ggml_backend_tensor_memset(mm,0,0,ggml_nbytes(mm));
    CHECK(ggml_backend_graph_compute(bridge,g)==GGML_STATUS_FAILED);
    for (float f:read(mm)) CHECK(f==0);
    ggml_backend_t backends[]={bridge,cpu};
    auto * sched = ggml_backend_sched_new(backends,nullptr,2,GGML_DEFAULT_GRAPH_SIZE,false,true); CHECK(sched);
    ggml_backend_sched_set_tensor_backend(sched,mm,bridge);
    ggml_backend_sched_set_tensor_backend(sched,y,cpu);
    CHECK(ggml_backend_sched_graph_compute(sched,g)==GGML_STATUS_SUCCESS);
    CHECK(ggml_backend_sched_get_tensor_backend(sched,mm)==bridge);
    CHECK(ggml_backend_sched_get_tensor_backend(sched,y)==cpu);
    compare(y,expected);
    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(buffer); ggml_free(ctx);
}

static void rejection_case(ggml_backend_t bridge) {
    auto * ctx = ggml_init({4*1024*1024,nullptr,true}); CHECK(ctx);
    auto dev=ggml_backend_get_device(bridge);
    auto * half_w=ggml_new_tensor_2d(ctx,GGML_TYPE_F16,8,4);
    auto * x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,8,3);
    CHECK(!ggml_backend_dev_supports_op(dev,ggml_mul_mat(ctx,half_w,x)));
    auto * a=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,4,4);
    auto * b=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,4,2);
    CHECK(!ggml_backend_dev_supports_op(dev,ggml_add(ctx,a,b))); // repeat != broadcast
    CHECK(!ggml_backend_dev_supports_op(dev,ggml_tanh_inplace(ctx,a)));
    auto * batched=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,4,4,2);
    CHECK(!ggml_backend_dev_supports_op(dev,ggml_tanh(ctx,batched)));
    ggml_free(ctx);
}

int main(int argc,char ** argv) {
    CHECK(argc==4);
    CHECK(ggml_backend_load(argv[2])); // CPU module from the same ggml build
    // Exercise the actual upstream out-of-tree discovery entry point.
#ifdef _WIN32
    _putenv_s("GGML_BACKEND_PATH",argv[1]);
#else
    setenv("GGML_BACKEND_PATH",argv[1],1);
#endif
    ggml_backend_load_all();
    auto dev=ggml_backend_dev_by_name(argv[3]); CHECK(dev);
    auto cpu=ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU,nullptr); CHECK(cpu);
    auto bridge=ggml_backend_dev_init(dev,nullptr); CHECK(bridge);
    rejection_case(bridge);
    graph_case(bridge,cpu,7,5,3,false,false);
    graph_case(bridge,cpu,7,5,3,false,true); // new addresses, same program signature
    graph_case(bridge,cpu,7,5,3,true,false); // offset, padded row strides
    graph_case(bridge,cpu,17,11,9,false,false);
    graph_case(bridge,cpu,7,5,1,false,false); // shape-cache replacement
    transpose_case(bridge,cpu);
    fallback_case(bridge,cpu);
    ggml_backend_free(bridge);
    bridge=ggml_backend_dev_init(dev,nullptr); CHECK(bridge);
    graph_case(bridge,cpu,7,5,3,false,false); // teardown and reopen
    ggml_backend_free(bridge); ggml_backend_free(cpu);
    std::printf("BRIDGE PASS: %s\n",argv[3]);
}
