#ifndef HTP_HMX_MM_GLU_H
#define HTP_HMX_MM_GLU_H

// Shared host / DSP plan of HTP_OP_MUL_MAT_NX_GLU: the gate / up MUL_MAT_NX (HMX 2-D, pipelined) with the
// split SWIGLU fused into its output epilogue. The gate and up jobs of one column chunk run back to back,
// so the layout is the pipelined HTP_MM_KERNEL_HMX_2D layout of htp_mm_hmx_vtcm_layout_build() plus two
// more fp16 output areas (gate / up of the next chunk) and per-thread f32 rows for the SWIGLU epilogue.
// The host solves (m_chunk, n_chunk, act threads) with htp_mm_glu_solve_2d_params(); the DSP rebuilds the
// layout from the same inputs with htp_mm_glu_vtcm_layout_build().

#include "matmul-ops.h"

#ifdef __cplusplus
extern "C" {
#endif

// f32 rows per thread for the epilogue: gate row pair, up row pair, sigmoid row (each n_chunk wide)
#define HTP_MM_GLU_EPI_ROWS 5

struct htp_mm_glu_vtcm_layout {
    size_t off_scales;
    size_t off_act;
    size_t off_act_f32;     // fp32 activation staging (overlaps group B during activation prep only)
    size_t off_weight[2];   // raw weight chunk from DMA: [0] gate, [1] up
    size_t off_scratch[2];  // dequantized fp16 weight chunk: [0] gate, [1] up
    size_t off_dst[4];      // fp16 HMX outputs: [0] gate, [1] up of even chunks; [2] gate, [3] up of odd chunks
    size_t off_epi;         // per-thread f32 epilogue rows

    size_t act_f32_bytes;
    size_t output_area_bytes;
    size_t epi_bytes_per_thread;

    size_t total_bytes;
};

static inline size_t htp_mm_glu_epi_bytes_per_thread(size_t nc) {
    return (size_t) HTP_MM_GLU_EPI_ROWS * nc * sizeof(float);
}

static inline void htp_mm_glu_vtcm_layout_build(struct htp_mm_glu_vtcm_layout * L,
                                                int                             wtype,
                                                uint32_t                        k,
                                                size_t                          mc,
                                                size_t                          nc,
                                                uint32_t                        act_threads,
                                                uint32_t                        aligned_tile_size,
                                                uint32_t                        n_threads) {
    const bool     is_quant     = (wtype != HTP_TYPE_F16 && wtype != HTP_TYPE_F32);
    const size_t   row_stride   = htp_mm_get_tiled_row_stride(wtype, k);
    const size_t   vec_dot_size = k * sizeof(uint16_t);
    const uint32_t n_k_tiles    = k / HTP_MM_HMX_TILE_N_COLS;

    const size_t min_f32_size     = hex_align_up(act_threads * HTP_MM_DMA_ACT_MULTIPLIER * k * sizeof(float), 128);
    const size_t weight_area_size = is_quant
        ? hex_align_up((nc / 32) * n_k_tiles * aligned_tile_size, HTP_MM_HMX_TILE_SIZE)
        : hex_align_up(nc * row_stride, HTP_MM_HMX_TILE_SIZE);
    const size_t act_area_size    = hex_align_up(mc * vec_dot_size, HTP_MM_HMX_TILE_SIZE);
    const size_t output_area_size = hex_align_up(mc * nc * sizeof(uint16_t), HTP_MM_HMX_TILE_SIZE);
    const size_t scratch_size     = hex_align_up(nc * vec_dot_size, HTP_MM_HMX_TILE_SIZE);
    const size_t epi_per_thread   = htp_mm_glu_epi_bytes_per_thread(nc);
    const size_t epi_size         = hex_align_up(n_threads * epi_per_thread, HTP_MM_HMX_TILE_SIZE);

    // Group A: scales and activation tiles (live for the whole M chunk)
    size_t off_group_a = 0;
    VTCM_LAYOUT_ALLOC(off_group_a, off_scales, HTP_MM_HMX_TILE_SIZE);
    VTCM_LAYOUT_ALLOC(off_group_a, off_act, act_area_size);

    // Group B: per-chunk buffers
    size_t off_group_b = off_group_a;
    VTCM_LAYOUT_ALLOC(off_group_b, off_weight[0], weight_area_size);
    VTCM_LAYOUT_ALLOC(off_group_b, off_weight[1], weight_area_size);
    VTCM_LAYOUT_ALLOC(off_group_b, off_dst[0], output_area_size);
    VTCM_LAYOUT_ALLOC(off_group_b, off_dst[1], output_area_size);
    VTCM_LAYOUT_ALLOC(off_group_b, off_dst[2], output_area_size);
    VTCM_LAYOUT_ALLOC(off_group_b, off_dst[3], output_area_size);
    VTCM_LAYOUT_ALLOC(off_group_b, off_scratch[0], scratch_size);
    VTCM_LAYOUT_ALLOC(off_group_b, off_scratch[1], scratch_size);
    VTCM_LAYOUT_ALLOC(off_group_b, off_epi, epi_size);

    const size_t group_b_size = off_group_b - off_group_a;

    // Group C: activation prep staging, overlaps group B (same rule as the 2-D layout)
    const size_t max_f32_size = act_threads * 64 * k * sizeof(float);
    const size_t act_f32_size = hex_align_up(hex_smin(max_f32_size, hex_smax(min_f32_size, group_b_size)), 128);
    size_t off_group_c = off_group_a;
    VTCM_LAYOUT_ALLOC(off_group_c, off_act_f32, act_f32_size);

    const size_t group_c_size = off_group_c - off_group_a;

    L->act_f32_bytes        = act_f32_size;
    L->output_area_bytes    = output_area_size;
    L->epi_bytes_per_thread = epi_per_thread;
    L->total_bytes          = off_group_a + hex_smax(group_b_size, group_c_size);
}

// Same search as htp_mm_hmx_solve_2d_params() (pipelined, no bias, not matmul_id) with the fused layout's
// per-column and per-element costs. Returns false when no plan fits the budget.
static inline bool htp_mm_glu_solve_2d_params(int      wtype,
                                              uint32_t k,
                                              uint32_t ne01_padded,
                                              uint32_t ne11,
                                              int      n_threads,
                                              uint32_t aligned_tile_size,
                                              size_t   vtcm_budget,
                                              size_t * m_chunk_out,
                                              size_t * n_chunk_out,
                                              int *    act_threads_out,
                                              size_t * vtcm_size_out) {
    const bool     is_quant      = (wtype != HTP_TYPE_F16 && wtype != HTP_TYPE_F32);
    const size_t   row_stride    = htp_mm_get_tiled_row_stride(wtype, k);
    const size_t   vec_dot_size  = k * sizeof(uint16_t);
    const uint32_t n_k_tiles     = k / HTP_MM_HMX_TILE_N_COLS;
    const size_t   qw_row_stride = is_quant ? (size_t) (n_k_tiles * aligned_tile_size) / 32 : 0;

    const size_t per_n  = 2 * (is_quant ? qw_row_stride : row_stride) + 2 * vec_dot_size +
                          (size_t) n_threads * HTP_MM_GLU_EPI_ROWS * sizeof(float);
    const size_t per_m  = vec_dot_size;
    const size_t per_mn = 4 * sizeof(uint16_t);
    // 9 regions rounded up to 2 KiB, plus the epilogue rows' rounding and the scales' slack
    const size_t overhead = 10 * HTP_MM_HMX_TILE_SIZE + 256;

    const uint32_t ne11_padded = hex_align_up(ne11, 32);

    size_t best_mblocks     = SIZE_MAX;
    int    best_act_threads = 0;
    size_t best_m_chunk     = 0;
    size_t best_n_chunk     = 0;
    size_t best_vtcm_size   = 0;

    int act_threads = n_threads;
    while (act_threads >= 1) {
        size_t mc = 0, nc = 0, total = 0;
        if (htp_mm_hmx_compute_chunks(vtcm_budget, overhead, per_n, per_m, per_mn, ne11_padded, ne01_padded,
                                      (size_t) ne01_padded * HTP_MM_HMX_COST_W_DEQUANT,
                                      (size_t) ne11 * HTP_MM_HMX_COST_A_CONVERT, &mc, &nc, &total) == 0) {
            struct htp_mm_glu_vtcm_layout L;
            htp_mm_glu_vtcm_layout_build(&L, wtype, k, mc, nc, (uint32_t) act_threads, aligned_tile_size, (uint32_t) n_threads);
            if (L.total_bytes <= vtcm_budget) {
                const size_t mblocks = ((size_t) ne11 + mc - 1) / mc;
                if (mblocks < best_mblocks || (mblocks == best_mblocks && act_threads > best_act_threads)) {
                    best_mblocks     = mblocks;
                    best_act_threads = act_threads;
                    best_m_chunk     = mc;
                    best_n_chunk     = nc;
                    best_vtcm_size   = L.total_bytes;
                }
            }
        }
        act_threads = (act_threads == 1) ? 0 : act_threads / 2;
    }

    if (best_act_threads == 0) {
        return false;
    }

    *m_chunk_out     = best_m_chunk;
    *n_chunk_out     = best_n_chunk;
    *act_threads_out = best_act_threads;
    *vtcm_size_out   = best_vtcm_size;
    return true;
}

#ifdef __cplusplus
}
#endif

#endif // HTP_HMX_MM_GLU_H
