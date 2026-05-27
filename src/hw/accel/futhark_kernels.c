// Stub for CPU-only builds (no Futhark CUDA backend)
// Provides all symbols declared in futhark_bindings.zig so the linker succeeds.
#include <stddef.h>
#include <stdint.h>

typedef struct futhark_context_config futhark_context_config;
typedef struct futhark_context futhark_context;
typedef struct futhark_f16_1d futhark_f16_1d;
typedef struct futhark_f16_2d futhark_f16_2d;
typedef struct futhark_f16_3d futhark_f16_3d;
typedef struct futhark_f32_1d futhark_f32_1d;
typedef struct futhark_f32_2d futhark_f32_2d;
typedef struct futhark_f32_3d futhark_f32_3d;
typedef struct futhark_u64_1d futhark_u64_1d;
typedef struct futhark_i64_1d futhark_i64_1d;

futhark_context_config* futhark_context_config_new(void) { return NULL; }
void futhark_context_config_free(futhark_context_config* cfg) { (void)cfg; }
void futhark_context_config_set_device(futhark_context_config* cfg, int device) { (void)cfg; (void)device; }
void futhark_context_config_set_platform(futhark_context_config* cfg, int platform) { (void)cfg; (void)platform; }
void futhark_context_config_set_default_group_size(futhark_context_config* cfg, int size) { (void)cfg; (void)size; }
void futhark_context_config_set_default_num_groups(futhark_context_config* cfg, int num) { (void)cfg; (void)num; }
void futhark_context_config_set_default_tile_size(futhark_context_config* cfg, int size) { (void)cfg; (void)size; }

futhark_context* futhark_context_new(futhark_context_config* cfg) { (void)cfg; return NULL; }
void futhark_context_free(futhark_context* ctx) { (void)ctx; }
int futhark_context_sync(futhark_context* ctx) { (void)ctx; return -1; }
const char* futhark_context_get_error(futhark_context* ctx) { (void)ctx; return "GPU not available (stub)"; }

futhark_f16_1d* futhark_new_f16_1d(futhark_context* ctx, const uint16_t* data, int64_t d0) { (void)ctx; (void)data; (void)d0; return NULL; }
futhark_f16_2d* futhark_new_f16_2d(futhark_context* ctx, const uint16_t* data, int64_t d0, int64_t d1) { (void)ctx; (void)data; (void)d0; (void)d1; return NULL; }
futhark_f16_3d* futhark_new_f16_3d(futhark_context* ctx, const uint16_t* data, int64_t d0, int64_t d1, int64_t d2) { (void)ctx; (void)data; (void)d0; (void)d1; (void)d2; return NULL; }
futhark_f16_2d* futhark_new_f16_2d_from_f32(futhark_context* ctx, const float* data, int64_t d0, int64_t d1) { (void)ctx; (void)data; (void)d0; (void)d1; return NULL; }
futhark_f16_3d* futhark_new_f16_3d_from_f32(futhark_context* ctx, const float* data, int64_t d0, int64_t d1, int64_t d2) { (void)ctx; (void)data; (void)d0; (void)d1; (void)d2; return NULL; }

int futhark_free_f16_1d(futhark_context* ctx, futhark_f16_1d* arr) { (void)ctx; (void)arr; return 0; }
int futhark_free_f16_2d(futhark_context* ctx, futhark_f16_2d* arr) { (void)ctx; (void)arr; return 0; }
int futhark_free_f16_3d(futhark_context* ctx, futhark_f16_3d* arr) { (void)ctx; (void)arr; return 0; }

int futhark_values_f16_1d(futhark_context* ctx, futhark_f16_1d* arr, uint16_t* data) { (void)ctx; (void)arr; (void)data; return -1; }
int futhark_values_f16_2d(futhark_context* ctx, futhark_f16_2d* arr, uint16_t* data) { (void)ctx; (void)arr; (void)data; return -1; }
int futhark_values_f16_3d(futhark_context* ctx, futhark_f16_3d* arr, uint16_t* data) { (void)ctx; (void)arr; (void)data; return -1; }
int futhark_values_f16_2d_to_f32(futhark_context* ctx, futhark_f16_2d* arr, float* data) { (void)ctx; (void)arr; (void)data; return -1; }
int futhark_values_f16_3d_to_f32(futhark_context* ctx, futhark_f16_3d* arr, float* data) { (void)ctx; (void)arr; (void)data; return -1; }

void* futhark_values_raw_f16_2d(futhark_context* ctx, futhark_f16_2d* arr) { (void)ctx; (void)arr; return NULL; }
int futhark_shape_f16_2d(futhark_context* ctx, futhark_f16_2d* arr, int64_t* dims) { (void)ctx; (void)arr; (void)dims; return -1; }

futhark_f32_1d* futhark_new_f32_1d(futhark_context* ctx, const float* data, int64_t d0) { (void)ctx; (void)data; (void)d0; return NULL; }
futhark_f32_2d* futhark_new_f32_2d(futhark_context* ctx, const float* data, int64_t d0, int64_t d1) { (void)ctx; (void)data; (void)d0; (void)d1; return NULL; }
futhark_f32_3d* futhark_new_f32_3d(futhark_context* ctx, const float* data, int64_t d0, int64_t d1, int64_t d2) { (void)ctx; (void)data; (void)d0; (void)d1; (void)d2; return NULL; }
futhark_u64_1d* futhark_new_u64_1d(futhark_context* ctx, const uint64_t* data, int64_t d0) { (void)ctx; (void)data; (void)d0; return NULL; }
futhark_i64_1d* futhark_new_i64_1d(futhark_context* ctx, const int64_t* data, int64_t d0) { (void)ctx; (void)data; (void)d0; return NULL; }

void futhark_free_f32_1d(futhark_context* ctx, futhark_f32_1d* arr) { (void)ctx; (void)arr; }
void futhark_free_f32_2d(futhark_context* ctx, futhark_f32_2d* arr) { (void)ctx; (void)arr; }
void futhark_free_f32_3d(futhark_context* ctx, futhark_f32_3d* arr) { (void)ctx; (void)arr; }
void futhark_free_u64_1d(futhark_context* ctx, futhark_u64_1d* arr) { (void)ctx; (void)arr; }
void futhark_free_i64_1d(futhark_context* ctx, futhark_i64_1d* arr) { (void)ctx; (void)arr; }

int futhark_values_f32_1d(futhark_context* ctx, futhark_f32_1d* arr, float* data) { (void)ctx; (void)arr; (void)data; return -1; }
int futhark_values_f32_2d(futhark_context* ctx, futhark_f32_2d* arr, float* data) { (void)ctx; (void)arr; (void)data; return -1; }
int futhark_values_f32_3d(futhark_context* ctx, futhark_f32_3d* arr, float* data) { (void)ctx; (void)arr; (void)data; return -1; }
int futhark_values_u64_1d(futhark_context* ctx, futhark_u64_1d* arr, uint64_t* data) { (void)ctx; (void)arr; (void)data; return -1; }
int futhark_values_i64_1d(futhark_context* ctx, futhark_i64_1d* arr, int64_t* data) { (void)ctx; (void)arr; (void)data; return -1; }

int futhark_entry_matmul(futhark_context* ctx, futhark_f32_2d** out, futhark_f32_2d* a, futhark_f32_2d* b) { (void)ctx; (void)out; (void)a; (void)b; return -1; }
int futhark_entry_batch_matmul(futhark_context* ctx, futhark_f32_3d** out, futhark_f32_3d* a, futhark_f32_3d* b) { (void)ctx; (void)out; (void)a; (void)b; return -1; }
int futhark_entry_dot(futhark_context* ctx, float* out, futhark_f32_1d* a, futhark_f32_1d* b) { (void)ctx; (void)out; (void)a; (void)b; return -1; }
int futhark_entry_clip_fisher(futhark_context* ctx, futhark_f32_1d** out, futhark_f32_1d* fisher, float clip_val) { (void)ctx; (void)out; (void)fisher; (void)clip_val; return -1; }
int futhark_entry_reduce_gradients(futhark_context* ctx, futhark_f32_1d** out, futhark_f32_2d* gradients) { (void)ctx; (void)out; (void)gradients; return -1; }
int futhark_entry_rank_segments(futhark_context* ctx, futhark_f32_1d** out, uint64_t query_hash, futhark_u64_1d* segment_hashes, futhark_f32_1d* base_scores) { (void)ctx; (void)out; (void)query_hash; (void)segment_hashes; (void)base_scores; return -1; }

int futhark_entry_rsf_forward(
    futhark_context* ctx, futhark_f16_2d** out,
    futhark_f16_2d* input, futhark_f16_2d* weights_s, futhark_f16_2d* weights_t,
    futhark_f16_1d* s_bias, futhark_f16_1d* t_bias,
    uint16_t clip_min, uint16_t clip_max) {
    (void)ctx; (void)out; (void)input; (void)weights_s; (void)weights_t;
    (void)s_bias; (void)t_bias; (void)clip_min; (void)clip_max;
    return -1;
}

int futhark_entry_rsf_backward(
    futhark_context* ctx,
    futhark_f16_2d** out_grad_ws, futhark_f16_2d** out_grad_wt,
    futhark_f16_1d** out_grad_sb, futhark_f16_1d** out_grad_tb,
    futhark_f16_2d* input, futhark_f16_2d* grad_output,
    futhark_f16_2d* weights_s, futhark_f16_2d* weights_t,
    futhark_f16_1d* s_bias, futhark_f16_1d* t_bias,
    uint16_t clip_min, uint16_t clip_max) {
    (void)ctx; (void)out_grad_ws; (void)out_grad_wt; (void)out_grad_sb; (void)out_grad_tb;
    (void)input; (void)grad_output; (void)weights_s; (void)weights_t;
    (void)s_bias; (void)t_bias; (void)clip_min; (void)clip_max;
    return -1;
}

int futhark_entry_scale_weights_inplace(futhark_context* ctx, futhark_f16_2d** out, futhark_f16_2d* weights, uint16_t scale) {
    (void)ctx; (void)out; (void)weights; (void)scale;
    return -1;
}

int futhark_entry_training_step(
    futhark_context* ctx,
    futhark_f16_2d** new_ws, futhark_f16_2d** new_wt,
    futhark_f16_1d** new_sb, futhark_f16_1d** new_tb,
    futhark_f16_2d** new_vs, futhark_f16_2d** new_vt,
    futhark_f16_1d** new_vsb, futhark_f16_1d** new_vtb,
    uint16_t* loss,
    futhark_f16_2d* inputs, futhark_f16_2d* targets,
    futhark_f16_2d* ws, futhark_f16_2d* wt,
    futhark_f16_1d* sb, futhark_f16_1d* tb,
    futhark_f16_2d* vs, futhark_f16_2d* vt,
    futhark_f16_1d* vsb, futhark_f16_1d* vtb,
    uint16_t lr, uint16_t momentum, uint16_t clip_min, uint16_t clip_max) {
    (void)ctx; (void)new_ws; (void)new_wt; (void)new_sb; (void)new_tb;
    (void)new_vs; (void)new_vt; (void)new_vsb; (void)new_vtb; (void)loss;
    (void)inputs; (void)targets; (void)ws; (void)wt; (void)sb; (void)tb;
    (void)vs; (void)vt; (void)vsb; (void)vtb;
    (void)lr; (void)momentum; (void)clip_min; (void)clip_max;
    return -1;
}
