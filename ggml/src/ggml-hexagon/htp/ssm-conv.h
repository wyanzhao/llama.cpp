#ifndef HTP_SSM_CONV_H
#define HTP_SSM_CONV_H

#include <stdint.h>

#include "hex-fastdiv.h"
#include "htp-ops.h"

struct htp_ssm_conv_kernel_params {
    uint32_t n_threads;
    uint32_t d_conv;
    uint32_t d_inner;
    uint32_t n_t;
    uint32_t n_s;
    uint32_t d_inner_per_thread;
    uint32_t d_inner_tile;

    uint32_t src0_row_size_aligned;
    uint32_t src1_row_size_aligned;
    uint32_t dst_row_size_aligned;

    uint32_t vtcm_src0_size_per_thread;
    uint32_t vtcm_src1_size_per_thread;
    uint32_t vtcm_dst_size_per_thread;

    uint32_t vtcm_src0_size;
    uint32_t vtcm_src1_size;
    uint32_t vtcm_dst_size;
    uint32_t vtcm_size;

    struct fastdiv_values div_n_threads;
};

// Fused CONCAT + CPY + SSM_CONV + SILU (HTP_OP_SSM_CONV_CHAIN), per sequence:
//   src[0] conv state {n_state, d_inner} (channel-major), src[1] qkv {n_t, d_inner} (transposed view), src[2] conv1d {d_conv, d_inner}
//   dst[0] silu(conv) {d_inner, n_t}, dst[1] new conv state {n_state * d_inner}
struct htp_ssm_conv_chain_kernel_params {
    uint32_t n_threads;
    uint32_t d_conv;
    uint32_t n_state;             // d_conv - 1
    uint32_t d_inner;
    uint32_t n_t;
    uint32_t ncs;                 // n_t + n_state, columns of the virtual concat
    uint32_t n_s;
    uint32_t d_inner_per_thread;  // multiple of 32
    uint32_t d_inner_tile;        // multiple of 32

    // byte strides between sequences
    uint32_t st_in_seq_stride;
    uint32_t qkv_seq_stride;
    uint32_t dst_seq_stride;
    uint32_t st_out_seq_stride;

    uint32_t n_buf;                   // tile buffers per thread
    uint32_t vtcm_xt_bytes;           // {ncs, d_inner_tile} conv input tile
    uint32_t vtcm_state_stage_bytes;  // conv state of the thread's channels, all sequences
    uint32_t vtcm_buf_stride;         // xt + state_out
    uint32_t vtcm_dst_tile_bytes;     // {n_t, d_inner_tile}

    uint32_t vtcm_src0_size_per_thread;
    uint32_t vtcm_src1_size_per_thread;
    uint32_t vtcm_dst_size_per_thread;

    uint32_t vtcm_src0_size;
    uint32_t vtcm_src1_size;
    uint32_t vtcm_dst_size;
    uint32_t vtcm_size;
};

#if defined(__cplusplus)
static_assert(sizeof(struct htp_ssm_conv_kernel_params) <= 128, "htp_ssm_conv_kernel_params is too large for kernel_params blob");
static_assert(sizeof(struct htp_ssm_conv_chain_kernel_params) <= 128, "htp_ssm_conv_chain_kernel_params is too large");
#else
_Static_assert(sizeof(struct htp_ssm_conv_kernel_params) <= 128, "htp_ssm_conv_kernel_params is too large for kernel_params blob");
_Static_assert(sizeof(struct htp_ssm_conv_chain_kernel_params) <= 128, "htp_ssm_conv_chain_kernel_params is too large");
#endif

#endif // HTP_SSM_CONV_H
