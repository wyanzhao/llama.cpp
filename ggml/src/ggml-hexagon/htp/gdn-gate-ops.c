#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"

#include <HAP_farf.h>
#include <math.h>
#include <string.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "dma-queue.h"
#include "hex-profile.h"
#include "htp-ops.h"
#include "htp-tensor.h"
#include "hvx-utils.h"
#include "hvx-sigmoid.h"
#include "unary-ops.h"

// GDN gate: dst = SOFTPLUS(src0 + src1) [* src2], optional dst[1] = SIGMOID(src3); src1, src2 are one row.
// Works on DDR with HVX loads and stores only: scalar loads leave L1 lines that the 128-byte-stride flush misses.
// src1/src2 are model weights: with DMA64 (v81+) weight buffers are mapped for DMA only, so their rows are
// staged in VTCM by DMA; the activations (src0, src3, dsts) are read and written in place.

struct htp_gdn_gate_context {
    struct htp_ops_context * octx;
    uint32_t                 nrows;
    uint32_t                 nrows_per_thread;
    uint32_t                 ne0;
    uint32_t                 row_slot;  // VTCM bytes per staged row, a multiple of 128
    int                      has_mul;
    int                      has_sigmoid;
};

// Load n bytes at p, touching only the aligned vectors that hold them.
static inline HVX_Vector gdn_gate_load(const void * p, uint32_t n) {
    const HVX_Vector * v  = (const HVX_Vector *) ((uintptr_t) p & ~(uintptr_t) 127);
    const HVX_Vector   v0 = v[0];
    const HVX_Vector   v1 = ((uintptr_t) p & 127) + n > 128 ? v[1] : v0;
    return Q6_V_valign_VVR(v1, v0, (size_t) p);
}

static void gdn_gate_thread(unsigned int nth, unsigned int ith, void * data) {
    (void) nth;

    struct htp_gdn_gate_context * gctx = (struct htp_gdn_gate_context *) data;
    struct htp_ops_context *      octx = gctx->octx;

    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * src1 = octx->src[1];
    const struct htp_tensor * src2 = octx->src[2];
    const struct htp_tensor * src3 = octx->src[3];
    const struct htp_tensor * dst  = octx->dsts[0];
    const struct htp_tensor * dst1 = octx->dsts[1];

    const uint32_t ne0 = gctx->ne0;
    const uint32_t ir0 = gctx->nrows_per_thread * ith;
    if (ir0 >= gctx->nrows) {
        return;
    }
    const uint32_t ir1 = MIN(ir0 + gctx->nrows_per_thread, gctx->nrows);

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    float __attribute__((aligned(128))) b_a[VLEN_FP32];
    float __attribute__((aligned(128))) b_b[VLEN_FP32];
    float __attribute__((aligned(128))) b_w[VLEN_FP32];
    float __attribute__((aligned(128))) b_o[VLEN_FP32];
    float __attribute__((aligned(128))) b_s[VLEN_FP32];

    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir0);

    // stage the one-row src1 (and src2) in this thread's VTCM slot
    const uint32_t row_nb = ne0 * sizeof(float);
    uint8_t *      vt     = octx->ctx->vtcm_base + (size_t) ith * 2 * gctx->row_slot;
    dma_queue *    dma_q  = octx->ctx->dma[ith];

    dma_queue_push(dma_q, dma_make_data(vt, src1->data), row_nb, row_nb, row_nb, 1);
    if (gctx->has_mul) {
        dma_queue_push(dma_q, dma_make_data(vt + gctx->row_slot, src2->data), row_nb, row_nb, row_nb, 1);
        dma_queue_pop(dma_q);
    }
    dma_queue_pop(dma_q);

    const float * bias = (const float *) vt;
    const float * w    = gctx->has_mul ? (const float *) (vt + gctx->row_slot) : NULL;

    for (uint32_t ir = ir0; ir < ir1; ++ir) {
        const size_t  off   = (size_t) ir * ne0 * sizeof(float);
        const float * a_row = (const float *) ((const uint8_t *) (uintptr_t) src0->data + off);
        float *       o_row = (float *) ((uint8_t *) (uintptr_t) dst->data + off);

        const float * s_row = gctx->has_sigmoid ?
                (const float *) ((const uint8_t *) (uintptr_t) src3->data + off) : NULL;
        float *       d_row = gctx->has_sigmoid ? (float *) ((uint8_t *) (uintptr_t) dst1->data + off) : NULL;

        for (uint32_t c = 0; c < ne0; c += VLEN_FP32) {
            const uint32_t cnt = MIN((uint32_t) VLEN_FP32, ne0 - c);
            const uint32_t nb  = cnt * sizeof(float);

            // read all operands of the chunk before writing any output (dst[1] may be src0)
            hvx_vmem(b_a) = gdn_gate_load(a_row + c, nb);
            if (gctx->has_sigmoid) {
                hvx_vmem(b_s) = gdn_gate_load(s_row + c, nb);
            }
            hvx_vmem(b_b) = gdn_gate_load(bias + c, nb);
            if (gctx->has_mul) {
                hvx_vmem(b_w) = gdn_gate_load(w + c, nb);
            }

            hvx_add_f32_aaa((uint8_t *) b_o, (const uint8_t *) b_a, (const uint8_t *) b_b, cnt);

            // same softplus as the standalone kernel
            for (uint32_t i = 0; i < cnt; ++i) {
                const float x = b_o[i];
                b_o[i] = (x > 20.0f) ? x : logf(1.0f + expf(x));
            }

            if (gctx->has_mul) {
                hvx_mul_f32_aaa((uint8_t *) b_o, (const uint8_t *) b_o, (const uint8_t *) b_w, cnt);
            }
            hvx_vec_store_u(o_row + c, nb, hvx_vmem(b_o));

            if (gctx->has_sigmoid) {
                hvx_sigmoid_f32_aa((uint8_t *) b_s, (const uint8_t *) b_s, cnt);
                hvx_vec_store_u(d_row + c, nb, hvx_vmem(b_s));
            }
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir1);
}

int op_gdn_gate(struct htp_ops_context * octx) {
    const struct htp_gdn_gate_kernel_params * kparams = (const struct htp_gdn_gate_kernel_params *) octx->kernel_params;

    if (octx->ctx->mdev.count > 1) {
        return HTP_STATUS_NO_SUPPORT;
    }
    if (kparams->nrows == 0) {
        return HTP_STATUS_OK;
    }
    if (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    struct htp_gdn_gate_context gctx = {
        .octx             = octx,
        .nrows            = kparams->nrows,
        .nrows_per_thread = kparams->nrows_per_thread,
        .ne0              = kparams->ne0,
        .row_slot         = hex_round_up(kparams->ne0 * sizeof(float), 128),
        .has_mul          = octx->src[2] != NULL,
        .has_sigmoid      = octx->src[3] != NULL,
    };

    if ((size_t) octx->n_threads * 2 * gctx.row_slot > octx->ctx->vtcm_size) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    FARF(HIGH, "gdn-gate-f32: %u rows of %u, threads %u mul %d sigmoid %d\n", gctx.nrows, gctx.ne0, octx->n_threads,
         gctx.has_mul, gctx.has_sigmoid);

    work_queue_run(octx->ctx->work_queue, gdn_gate_thread, &gctx, octx->n_threads);

    return HTP_STATUS_OK;
}
