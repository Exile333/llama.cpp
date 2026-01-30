#pragma once

#include "common.cuh"
#include "vecdotq.cuh"
#include "vecdotq_preloaded_data.cuh"

// TODO Merge with "vecdotq_preloaded_data.cuh".

////////// Q8_1 preload stuff.

static __device__ __forceinline__ void y_q4_0_preloader(const block_q8_1 * __restrict__ y, uint8_t * __restrict__ result, const int& kby, const int& kqs) {
    preloaded_data_q4_0_q8_1 * preloaded_data = (preloaded_data_q4_0_q8_1 *) result;

#pragma unroll
    for (int i = 0; i < VDR_Q4_0_Q8_1_MMVQ; ++i) {
        preloaded_data->scales_q8_1[2*i+0] = get_int_b4(y[kby].qs, kqs + i);
        preloaded_data->scales_q8_1[2*i+1] = get_int_b4(y[kby].qs, kqs + i + QI4_0);
    }
    preloaded_data->ds_q8_1 = y[kby].ds;
}

static __device__ __forceinline__ void y_q6_K_preloader(const block_q8_1 * __restrict__ y, uint8_t * __restrict__ result, const int& kby, const int& kqs) {
    preloaded_data_q6_K_q8_1 * preloaded_data = (preloaded_data_q6_K_q8_1 *) result;
    const int y_offset = 2 * QR6_K * (kqs / (QI6_K/2)) + (kqs % (QI6_K/2)) / (QI6_K/4);

#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        preloaded_data->scales_q8_1[i] = get_int_b4(y[kby + y_offset + 2*i].qs, kqs % QI8_1);
        preloaded_data->ds_q8_1[i] = __low2float(y[kby + y_offset + 2*i].ds);
    }
}

static __device__ __forceinline__ void y_q4_1_preloader(const block_q8_1 * __restrict__ y, uint8_t * __restrict__ result, const int& kby, const int& kqs) {
    preloaded_data_q4_1_q8_1 * preloaded_data = (preloaded_data_q4_1_q8_1 *) result;

#pragma unroll
    for (int i = 0; i < VDR_Q4_1_Q8_1_MMVQ; ++i) {
        preloaded_data->scales_q8_1[2*i+0] = get_int_b4(y[kby].qs, kqs + i);
        preloaded_data->scales_q8_1[2*i+1] = get_int_b4(y[kby].qs, kqs + i + QI4_1);
    }
    preloaded_data->ds_q8_1 = y[kby].ds;
}

static __device__ __forceinline__ void y_q5_0_preloader(const block_q8_1 * __restrict__ y, uint8_t * __restrict__ result, const int& kby, const int& kqs) {
    preloaded_data_q5_0_q8_1 * preloaded_data = (preloaded_data_q5_0_q8_1 *) result;

#pragma unroll
    for (int i = 0; i < VDR_Q5_0_Q8_1_MMVQ; ++i) {
        preloaded_data->scales_q8_1[2*i+0] = get_int_b4(y[kby].qs, kqs + i);
        preloaded_data->scales_q8_1[2*i+1] = get_int_b4(y[kby].qs, kqs + i + QI5_0);
    }
    preloaded_data->ds_q8_1 = y[kby].ds;
}

static __device__ __forceinline__ void y_q5_1_preloader(const block_q8_1 * __restrict__ y, uint8_t * __restrict__ result, const int& kby, const int& kqs) {
    preloaded_data_q5_1_q8_1 * preloaded_data = (preloaded_data_q5_1_q8_1 *) result;

#pragma unroll
    for (int i = 0; i < VDR_Q5_1_Q8_1_MMVQ; ++i) {
        preloaded_data->scales_q8_1[2*i+0] = get_int_b4(y[kby].qs, kqs + i);
        preloaded_data->scales_q8_1[2*i+1] = get_int_b4(y[kby].qs, kqs + i + QI5_1);
    }
    preloaded_data->ds_q8_1 = y[kby].ds;
}

static __device__ __forceinline__ void y_q8_0_preloader(const block_q8_1 * __restrict__ y, uint8_t * __restrict__ result, const int& kby, const int& kqs) {
    preloaded_data_q8_0_q8_1 * preloaded_data = (preloaded_data_q8_0_q8_1 *) result;

#pragma unroll
    for (int i = 0; i < VDR_Q8_0_Q8_1_MMVQ; ++i) {
        preloaded_data->scales_q8_1[i] = get_int_b4(y[kby].qs, kqs + i);
    }
    preloaded_data->d_q8_1 = __low2half(y[kby].ds);
}

static __device__ __forceinline__ void y_mxfp4_preloader(const block_q8_1 * __restrict__ y, uint8_t * __restrict__ result, const int& kby, const int& kqs) {
    preloaded_data_mxfp4_q8_1 * preloaded_data = (preloaded_data_mxfp4_q8_1 *) result;

#pragma unroll
    for (int i = 0; i < VDR_MXFP4_Q8_1_MMVQ; ++i) {
        preloaded_data->u_q8_1[i*2 + 0] = get_int_b4(y[kby].qs, kqs + i);
        preloaded_data->u_q8_1[i*2 + 1] = get_int_b4(y[kby].qs, kqs + i + 4);
    }
    preloaded_data->ds_q8_1 = y[kby].ds;
}

static __device__ __forceinline__ void y_q2_K_preloader(const block_q8_1 * __restrict__ y, uint8_t * __restrict__ result, const int& kby, const int& kqs) {
    preloaded_data_q2_K_q8_1 * preloaded_data = (preloaded_data_q2_K_q8_1 *) result;
    const int bq8_offset = QR2_K * (kqs / QI8_1);

#pragma unroll
    for (int i = 0; i < QR2_K; ++i) {
        preloaded_data->scales_q8_1[i] = get_int_b4(y[kby + bq8_offset + i].qs, kqs % QI8_1);
        preloaded_data->ds_q8_1[i] = __low2float(y[kby + bq8_offset + i].ds);
    }
}

static __device__ __forceinline__ void y_q3_K_preloader(const block_q8_1 * __restrict__ y, uint8_t * __restrict__ result, const int& kby, const int& kqs) {
    preloaded_data_q3_K_q8_1 * preloaded_data = (preloaded_data_q3_K_q8_1 *) result;
    const int bq8_offset = QR3_K * (kqs / (QI3_K/2));

#pragma unroll
    for (int i = 0; i < QR3_K; ++i) {
        preloaded_data->scales_q8_1[i] = get_int_b4(y[kby + bq8_offset + i].qs, kqs % QI8_1);
        preloaded_data->ds_q8_1[i] = __low2float(y[kby + bq8_offset + i].ds);
    }
}

static __device__ __forceinline__ void y_q4_K_preloader(const block_q8_1 * __restrict__ y, uint8_t * __restrict__ result, const int& kby, const int& kqs) {
    preloaded_data_q4_K_q8_1 * preloaded_data = (preloaded_data_q4_K_q8_1 *) result;

    const int bq8_offset = QR4_K * ((kqs/2) / (QI8_1/2));
#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const block_q8_1 * bq8i = y + kby + bq8_offset + i;
        preloaded_data->d8[i] = bq8i->ds;

        const int * q8 = (const int *)bq8i->qs + ((kqs/2)%4);
        preloaded_data->u[2*i+0] = q8[0];
        preloaded_data->u[2*i+1] = q8[4];
    }
}

static __device__ __forceinline__ void y_q5_K_preloader(const block_q8_1 * __restrict__ y, uint8_t * __restrict__ result, const int& kby, const int& kqs) {
    preloaded_data_q5_K_q8_1 * preloaded_data = (preloaded_data_q5_K_q8_1 *) result;
    const int bq8_offset = QR5_K * ((kqs/2) / (QI8_1/2));

#pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const block_q8_1 * bq8i = y + kby + bq8_offset + i;
        preloaded_data->ds_q8_1[i] = __low2float(bq8i->ds);

        const int * q8 = (const int *)bq8i->qs + ((kqs/2)%4);
        preloaded_data->scales_q8_1[2*i+0] = q8[0];
        preloaded_data->scales_q8_1[2*i+1] = q8[4];
    }
}

static __device__ __forceinline__ void y_iq2_xxs_preloader(const block_q8_1 * __restrict__ y, uint8_t * __restrict__ result, const int& kby, const int& kqs) {
    preloaded_data_iq2_xxs_q8_1 * preloaded_data = (preloaded_data_iq2_xxs_q8_1 *) result;

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        preloaded_data->scales_q8_1[i] = get_int_b4(y[kby + kqs/2].qs, i);
    }
    preloaded_data->d_q8_1 = __low2half(y[kby + kqs/2].ds);
}

static __device__ __forceinline__ void y_iq2_xs_preloader(const block_q8_1 * __restrict__ y, uint8_t * __restrict__ result, const int& kby, const int& kqs) {
    preloaded_data_iq2_xs_q8_1 * preloaded_data = (preloaded_data_iq2_xs_q8_1 *) result;

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        preloaded_data->scales_q8_1[i] = get_int_b4(y[kby + kqs/2].qs, i);
    }
    preloaded_data->d_q8_1 = __low2half(y[kby + kqs/2].ds);
}

static __device__ __forceinline__ void y_iq2_s_preloader(const block_q8_1 * __restrict__ y, uint8_t * __restrict__ result, const int& kby, const int& kqs) {
    preloaded_data_iq2_s_q8_1 * preloaded_data = (preloaded_data_iq2_s_q8_1 *) result;

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        preloaded_data->scales_q8_1[i] = get_int_b4(y[kby + kqs/2].qs, i);
    }
    preloaded_data->d_q8_1 = __low2half(y[kby + kqs/2].ds);
}

static __device__ __forceinline__ void y_iq3_xxs_preloader(const block_q8_1 * __restrict__ y, uint8_t * __restrict__ result, const int& kby, const int& kqs) {
    preloaded_data_iq3_xxs_q8_1 * preloaded_data = (preloaded_data_iq3_xxs_q8_1 *) result;

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        preloaded_data->scales_q8_1[i] = get_int_b4(y[kby + kqs/2].qs, i);
    }
    preloaded_data->d_q8_1 = __low2half(y[kby + kqs/2].ds);
}

static __device__ __forceinline__ void y_iq3_s_preloader(const block_q8_1 * __restrict__ y, uint8_t * __restrict__ result, const int& kby, const int& kqs) {
    preloaded_data_iq3_s_q8_1 * preloaded_data = (preloaded_data_iq3_s_q8_1 *) result;

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        preloaded_data->scales_q8_1[i] = get_int_b4(y[kby + kqs/2].qs, i);
    }
    preloaded_data->d_q8_1 = __low2half(y[kby + kqs/2].ds);
}

static __device__ __forceinline__ void y_iq1_s_preloader(const block_q8_1 * __restrict__ y, uint8_t * __restrict__ result, const int& kby, const int& kqs) {
    preloaded_data_iq1_s_q8_1 * preloaded_data = (preloaded_data_iq1_s_q8_1 *) result;

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        preloaded_data->scales_q8_1[i] = get_int_b4(y[kby + kqs].qs, i);
    }
    preloaded_data->ds_q8_1 = y[kby + kqs].ds;
}

static __device__ __forceinline__ void y_iq1_m_preloader(const block_q8_1 * __restrict__ y, uint8_t * __restrict__ result, const int& kby, const int& kqs) {
    preloaded_data_iq1_m_q8_1 * preloaded_data = (preloaded_data_iq1_m_q8_1 *) result;

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        preloaded_data->scales_q8_1[i] = get_int_b4(y[kby + kqs].qs, i);
    }
}

static __device__ __forceinline__ void y_iq4_nl_preloader(const block_q8_1 * __restrict__ y, uint8_t * __restrict__ result, const int& kby, const int& kqs) {
    preloaded_data_iq4_nl_q8_1 * preloaded_data = (preloaded_data_iq4_nl_q8_1 *) result;

#pragma unroll
    for (int i = 0; i < VDR_IQ4_NL_Q8_1_MMVQ + 4; ++i) {
        preloaded_data->scales_q8_1[i] = get_int_b4(y[kby].qs, kqs + i);
    }
    preloaded_data->d_q8_1 = __low2half(y[kby].ds);
}

static __device__ __forceinline__ void y_iq4_xs_preloader(const block_q8_1 * __restrict__ y, uint8_t * __restrict__ result, const int& kby, const int& kqs) {
    preloaded_data_iq4_xs_q8_1 * preloaded_data = (preloaded_data_iq4_xs_q8_1 *) result;

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        preloaded_data->scales_q8_1[i] = get_int_b4(y[kby + kqs/4].qs, i);
    }
    preloaded_data->d_q8_1 = __low2half(y[kby + kqs/4].ds);
}

///////////////////// x preload stuff

static constexpr __device__ int get_preloaded_data_size(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:    return sizeof(preloaded_data_q4_0_q8_1);
        case GGML_TYPE_Q4_1:    return sizeof(preloaded_data_q4_1_q8_1);
        case GGML_TYPE_Q5_0:    return sizeof(preloaded_data_q5_0_q8_1);
        case GGML_TYPE_Q5_1:    return sizeof(preloaded_data_q5_1_q8_1);
        case GGML_TYPE_Q8_0:    return sizeof(preloaded_data_q8_0_q8_1);
        case GGML_TYPE_MXFP4:   return sizeof(preloaded_data_mxfp4_q8_1);
        case GGML_TYPE_Q2_K:    return sizeof(preloaded_data_q2_K_q8_1);
        case GGML_TYPE_Q3_K:    return sizeof(preloaded_data_q3_K_q8_1);
        case GGML_TYPE_Q4_K:    return sizeof(preloaded_data_q4_K_q8_1);
        case GGML_TYPE_Q5_K:    return sizeof(preloaded_data_q5_K_q8_1);
        case GGML_TYPE_Q6_K:    return sizeof(preloaded_data_q6_K_q8_1);
        case GGML_TYPE_IQ2_XXS: return sizeof(preloaded_data_iq2_xxs_q8_1);
        case GGML_TYPE_IQ2_XS:  return sizeof(preloaded_data_iq2_xs_q8_1);
        case GGML_TYPE_IQ2_S:   return sizeof(preloaded_data_iq2_s_q8_1);
        case GGML_TYPE_IQ3_XXS: return sizeof(preloaded_data_iq3_xxs_q8_1);
        case GGML_TYPE_IQ3_S:   return sizeof(preloaded_data_iq3_s_q8_1);
        case GGML_TYPE_IQ1_S:   return sizeof(preloaded_data_iq1_s_q8_1);
        case GGML_TYPE_IQ1_M:   return sizeof(preloaded_data_iq1_m_q8_1);
        case GGML_TYPE_IQ4_NL:  return sizeof(preloaded_data_iq4_nl_q8_1);
        case GGML_TYPE_IQ4_XS:  return sizeof(preloaded_data_iq4_xs_q8_1);
        default:                return 0;
    }
}

static __device__ __forceinline__ void x_q4_0_preloader(const void * __restrict__ vx, uint8_t * __restrict__ result, const int& kbx, const int& kqs) {
    preloaded_data_q4_0_q8_1 * preloaded_data = (preloaded_data_q4_0_q8_1 *) result;
    const block_q4_0 * bq4_0 = (const block_q4_0 *) vx + kbx;

#pragma unroll
    for (int i = 0; i < VDR_Q4_0_Q8_1_MMVQ; ++i) {
        preloaded_data->scales_q4_0[i] = get_int_b2(bq4_0->qs, kqs + i);
    }
    preloaded_data->d_q4_0 = bq4_0->d;
}

static __device__ __forceinline__ void x_q6_K_preloader(const void * __restrict__ vx, uint8_t * __restrict__ result, const int& kbx, const int& kqs) {
    preloaded_data_q6_K_q8_1 * preloaded_data = (preloaded_data_q6_K_q8_1 *) result;
    const int scale_offset = (QI6_K/4) * (kqs / (QI6_K/2)) + (kqs % (QI6_K/2)) / (QI6_K/8);
    const int vh_shift = 2 * ((kqs % (QI6_K/2)) / (QI6_K/4));

    const block_q6_K * bq6_K = (const block_q6_K *) vx + kbx;

    preloaded_data->vl = get_int_b2(bq6_K->ql, kqs);
    preloaded_data->vh = get_int_b2(bq6_K->qh, (QI6_K/4) * (kqs / (QI6_K/2)) + kqs % (QI6_K/4)) >> vh_shift;
#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        preloaded_data->scales_q6_K[i] = bq6_K->scales[scale_offset + 4*i];
    }
    preloaded_data->d_q6_K = bq6_K->d;
}

static __device__ __forceinline__ void x_q4_1_preloader(const void * __restrict__ vx, uint8_t * __restrict__ result, const int& kbx, const int& kqs) {
    preloaded_data_q4_1_q8_1 * preloaded_data = (preloaded_data_q4_1_q8_1 *) result;
    const block_q4_1 * bq4_1 = (const block_q4_1 *) vx + kbx;

#pragma unroll
    for (int i = 0; i < VDR_Q4_1_Q8_1_MMVQ; ++i) {
        preloaded_data->scales_q4_1[i] = get_int_b4(bq4_1->qs, kqs + i);
    }
    preloaded_data->dm_q4_1 = bq4_1->dm;
}

static __device__ __forceinline__ void x_q5_0_preloader(const void * __restrict__ vx, uint8_t * __restrict__ result, const int& kbx, const int& kqs) {
    preloaded_data_q5_0_q8_1 * preloaded_data = (preloaded_data_q5_0_q8_1 *) result;
    const block_q5_0 * bq5_0 = (const block_q5_0 *) vx + kbx;

#pragma unroll
    for (int i = 0; i < VDR_Q5_0_Q8_1_MMVQ; ++i) {
        preloaded_data->vl[i] = get_int_b2(bq5_0->qs, kqs + i);
        preloaded_data->vh[i] = get_int_b2(bq5_0->qh, 0) >> (4 * (kqs + i));
    }
    preloaded_data->d_q5_0 = bq5_0->d;
}

static __device__ __forceinline__ void x_q5_1_preloader(const void * __restrict__ vx, uint8_t * __restrict__ result, const int& kbx, const int& kqs) {
    preloaded_data_q5_1_q8_1 * preloaded_data = (preloaded_data_q5_1_q8_1 *) result;
    const block_q5_1 * bq5_1 = (const block_q5_1 *) vx + kbx;

#pragma unroll
    for (int i = 0; i < VDR_Q5_1_Q8_1_MMVQ; ++i) {
        preloaded_data->vl[i] = get_int_b4(bq5_1->qs, kqs + i);
        preloaded_data->vh[i] = get_int_b4(bq5_1->qh, 0) >> (4 * (kqs + i));
    }
    preloaded_data->dm_q5_1 = bq5_1->dm;
}

static __device__ __forceinline__ void x_q8_0_preloader(const void * __restrict__ vx, uint8_t * __restrict__ result, const int& kbx, const int& kqs) {
    preloaded_data_q8_0_q8_1 * preloaded_data = (preloaded_data_q8_0_q8_1 *) result;
    const block_q8_0 * bq8_0 = (const block_q8_0 *) vx + kbx;

#pragma unroll
    for (int i = 0; i < VDR_Q8_0_Q8_1_MMVQ; ++i) {
        preloaded_data->scales_q8_0[i] = get_int_b2(bq8_0->qs, kqs + i);
    }
    preloaded_data->d_q8_0 = bq8_0->d;
}

static __device__ __forceinline__ void x_mxfp4_preloader(const void * __restrict__ vx, uint8_t * __restrict__ result, const int& kbx, const int& kqs) {
    preloaded_data_mxfp4_q8_1 * preloaded_data = (preloaded_data_mxfp4_q8_1 *) result;
    const block_mxfp4 * bq4 = (const block_mxfp4 *) vx + kbx;

#pragma unroll
    for (int l = 0; l < VDR_MXFP4_Q8_1_MMVQ; ++l) {
        preloaded_data->aux_q4[l] = get_int_b1(bq4->qs, kqs + l);
    }
    preloaded_data->e_mxfp4 = bq4->e;
}

static __device__ __forceinline__ void x_q2_K_preloader(const void * __restrict__ vx, uint8_t * __restrict__ result, const int& kbx, const int& kqs) {
    preloaded_data_q2_K_q8_1 * preloaded_data = (preloaded_data_q2_K_q8_1 *) result;
    const block_q2_K * bq2_K = (const block_q2_K *) vx + kbx;

    const int scale_offset = kqs - kqs % QI8_1 + (kqs % QI8_1) / (QI8_1/2);

    preloaded_data->v = get_int_b4(bq2_K->qs, kqs);
#pragma unroll
    for (int i = 0; i < 2*QR2_K; ++i) {
        preloaded_data->scales_q2_K[i] = bq2_K->scales[scale_offset + i];
    }
    preloaded_data->dm_q2_K = bq2_K->dm;
}

static __device__ __forceinline__ void x_q3_K_preloader(const void * __restrict__ vx, uint8_t * __restrict__ result, const int& kbx, const int& kqs) {
    preloaded_data_q3_K_q8_1 * preloaded_data = (preloaded_data_q3_K_q8_1 *) result;
    const block_q3_K * bq3_K = (const block_q3_K *) vx + kbx;

    const int bq8_offset = QR3_K * (kqs / (QI3_K/2));
    const int scale_offset = kqs - kqs % QI8_1 + (kqs % QI8_1) / (QI8_1/2);

    preloaded_data->vl = get_int_b2(bq3_K->qs, kqs);
    preloaded_data->vh = ~get_int_b2(bq3_K->hmask, kqs % (QI3_K/2)) >> bq8_offset;
#pragma unroll
    for (int i = 0; i < (QK_K/32) + (QK_K/64); ++i) {
        preloaded_data->scales_q3_K[i] = bq3_K->scales[i];
    }
    preloaded_data->scale_offset = scale_offset;
    preloaded_data->d_q3_K = bq3_K->d;
}

static __device__ __forceinline__ void x_q4_K_preloader(const void * __restrict__ vx, uint8_t * __restrict__ result, const int& kbx, const int& kqs) {
    preloaded_data_q4_K_q8_1 * preloaded_data = (preloaded_data_q4_K_q8_1 *) result;
    const block_q4_K * bq4_K = (const block_q4_K *) vx + kbx;

    const int bq8_offset = QR4_K * ((kqs/2) / (QI8_1/2));

    const int * q4 = (const int *)(bq4_K->qs + 16 * bq8_offset + 4 * ((kqs/2)%4));
    preloaded_data->v[0] = q4[0];
    preloaded_data->v[1] = q4[4];

    {
        const uint16_t * scales = (const uint16_t *)bq4_K->scales;
        const int j = bq8_offset/2;
        preloaded_data->q4_K_scm[0] = scales[j+0];
        preloaded_data->q4_K_scm[1] = scales[j+2];
        preloaded_data->q4_K_scm[2] = j < 2 ? 0 : scales[j-2];
        if (j < 2) {
            preloaded_data->q4_K_scm_extended = false;
        } else {
            preloaded_data->q4_K_scm_extended = true;
        }
    }

    preloaded_data->dm_q4_K = bq4_K->dm;
}

static __device__ __forceinline__ void x_q5_K_preloader(const void * __restrict__ vx, uint8_t * __restrict__ result, const int& kbx, const int& kqs) {
    preloaded_data_q5_K_q8_1 * preloaded_data = (preloaded_data_q5_K_q8_1 *) result;
    const block_q5_K * bq5_K = (const block_q5_K *) vx + kbx;

    const int bq8_offset = QR5_K * ((kqs/2) / (QI8_1/2));
    const int * ql = (const int *)(bq5_K->qs + 16 * bq8_offset + 4 * ((kqs/2)%4));
    const int * qh = (const int *)(bq5_K->qh + 4 * ((kqs/2)%4));

    preloaded_data->vl[0] = ql[0];
    preloaded_data->vl[1] = ql[4];

    preloaded_data->vh[0] = qh[0] >> bq8_offset;
    preloaded_data->vh[1] = qh[4] >> bq8_offset;

    const uint16_t * scales = (const uint16_t *)bq5_K->scales;
    uint16_t aux[2];
    const int j = bq8_offset/2;
    if (j < 2) {
        aux[0] = scales[j+0] & 0x3f3f;
        aux[1] = scales[j+2] & 0x3f3f;
    } else {
        aux[0] = ((scales[j+2] >> 0) & 0x0f0f) | ((scales[j-2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j+2] >> 4) & 0x0f0f) | ((scales[j-0] & 0xc0c0) >> 2);
    }
    const uint8_t * sc = (const uint8_t *)aux;
    const uint8_t * m  = sc + 2;

#pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        preloaded_data->sc[i] = sc[i];
        preloaded_data->m[i] = m[i];
    }
    preloaded_data->dm_q5_K = bq5_K->dm;
}

static __device__ __forceinline__ void x_iq2_xxs_preloader(const void * __restrict__ vx, uint8_t * __restrict__ result, const int& kbx, const int& kqs) {
    preloaded_data_iq2_xxs_q8_1 * preloaded_data = (preloaded_data_iq2_xxs_q8_1 *) result;
    const block_iq2_xxs * bq2 = (const block_iq2_xxs *) vx + kbx;

    preloaded_data->q2 = get_int_b2(bq2->qs, kqs);
    preloaded_data->aux32 = get_int_b2(bq2->qs, kqs + 1);
    preloaded_data->d_iq2_xxs = bq2->d;
}

static __device__ __forceinline__ void x_iq2_xs_preloader(const void * __restrict__ vx, uint8_t * __restrict__ result, const int& kbx, const int& kqs) {
    preloaded_data_iq2_xs_q8_1 * preloaded_data = (preloaded_data_iq2_xs_q8_1 *) result;
    const block_iq2_xs * bq2 = (const block_iq2_xs *) vx + kbx;

    preloaded_data->q2_packed = make_int2(get_int_b2(bq2->qs, kqs + 0), get_int_b2(bq2->qs, kqs + 1));
    preloaded_data->ls0 = bq2->scales[kqs/2] & 0x0F;
    preloaded_data->ls1 = bq2->scales[kqs/2] >> 4;
    preloaded_data->d_iq2_xs = bq2->d;
}

static __device__ __forceinline__ void x_iq2_s_preloader(const void * __restrict__ vx, uint8_t * __restrict__ result, const int& kbx, const int& kqs) {
    preloaded_data_iq2_s_q8_1 * preloaded_data = (preloaded_data_iq2_s_q8_1 *) result;
    const block_iq2_s * bq2 = (const block_iq2_s *) vx + kbx;

    preloaded_data->qs_packed = get_int_b2(bq2->qs, kqs/2);
    preloaded_data->qh = bq2->qh[kqs/2];
    preloaded_data->signs_packed_32 = get_int_b2(bq2->qs, QK_K/32 + kqs/2);
    preloaded_data->ls0 = bq2->scales[kqs/2] & 0x0F;
    preloaded_data->ls1 = bq2->scales[kqs/2] >> 4;
    preloaded_data->d_iq2_s = bq2->d;
}

static __device__ __forceinline__ void x_iq3_xxs_preloader(const void * __restrict__ vx, uint8_t * __restrict__ result, const int& kbx, const int& kqs) {
    preloaded_data_iq3_xxs_q8_1 * preloaded_data = (preloaded_data_iq3_xxs_q8_1 *) result;
    const block_iq3_xxs * bq3 = (const block_iq3_xxs *) vx + kbx;

    preloaded_data->q3_packed = make_int2(get_int_b2(bq3->qs, kqs), get_int_b2(bq3->qs, kqs+1));
    preloaded_data->aux32 = get_int_b2(bq3->qs, QK_K/16 + kqs/2);
    preloaded_data->d_iq3_xxs = bq3->d;
}

static __device__ __forceinline__ void x_iq3_s_preloader(const void * __restrict__ vx, uint8_t * __restrict__ result, const int& kbx, const int& kqs) {
    preloaded_data_iq3_s_q8_1 * preloaded_data = (preloaded_data_iq3_s_q8_1 *) result;
    const block_iq3_s * bq3 = (const block_iq3_s *) vx + kbx;

    preloaded_data->qs_packed = make_int2(get_int_b2(bq3->qs, kqs + 0), get_int_b2(bq3->qs, kqs + 1));
    preloaded_data->qh = bq3->qh[kqs/2];
    preloaded_data->signs_packed_32 = get_int_b2(bq3->signs, kqs/2);
    preloaded_data->scale_val = 1 + 2*((bq3->scales[kqs/4] >> ((kqs << 1) & 0x04)) & 0x0F);
    preloaded_data->d_iq3_s = bq3->d;
}

static __device__ __forceinline__ void x_iq1_s_preloader(const void * __restrict__ vx, uint8_t * __restrict__ result, const int& kbx, const int& kqs) {
    preloaded_data_iq1_s_q8_1 * preloaded_data = (preloaded_data_iq1_s_q8_1 *) result;
    const block_iq1_s * bq1 = (const block_iq1_s *) vx + kbx;

    preloaded_data->qs_packed = get_int_b2(bq1->qs, kqs);
    preloaded_data->qh = bq1->qh[kqs];
    preloaded_data->d1q = __half2float(bq1->d) * (((bq1->qh[kqs] >> 11) & 0x0E) + 1);
    preloaded_data->delta = -1.0f + IQ1S_DELTA - (bq1->qh[kqs] & 0x8000) * (2.0f*IQ1S_DELTA/0x8000);
}

static __device__ __forceinline__ void x_iq1_m_preloader(const void * __restrict__ vx, uint8_t * __restrict__ result, const int& kbx, const int& kqs) {
    preloaded_data_iq1_m_q8_1 * preloaded_data = (preloaded_data_iq1_m_q8_1 *) result;
    const block_iq1_m * bq1 = (const block_iq1_m *) vx + kbx;

    preloaded_data->qs_packed = get_int_b4(bq1->qs, kqs);
    preloaded_data->qh[0] = bq1->qh[2*kqs + 0] >> 0;
    preloaded_data->qh[1] = bq1->qh[2*kqs + 1] >> 4;

    const uint16_t * sc = (const uint16_t *) bq1->scales;

    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
    preloaded_data->d = __half2float(scale.f16);

    const int tmp = sc[kqs/2] >> (6*(kqs%2));
    preloaded_data->sc0 = 2*((tmp >> 0) & 0x07) + 1;
    preloaded_data->sc1 = 2*((tmp >> 3) & 0x07) + 1;
}

static __device__ __forceinline__ void x_iq4_nl_preloader(const void * __restrict__ vx, uint8_t * __restrict__ result, const int& kbx, const int& kqs) {
    preloaded_data_iq4_nl_q8_1 * preloaded_data = (preloaded_data_iq4_nl_q8_1 *) result;
    const block_iq4_nl * bq4 = (const block_iq4_nl *) vx + kbx;

#pragma unroll
    for (int l = 0; l < VDR_IQ4_NL_Q8_1_MMVQ; ++l) {
        const int aux_q4 = get_int_b2(bq4->qs, kqs + l);
        preloaded_data->v_table[l] = get_int_from_table_16(aux_q4, kvalues_iq4nl);
    }
    preloaded_data->d_iq4_nl = bq4->d;
}

static __device__ __forceinline__ void x_iq4_xs_preloader(const void * __restrict__ vx, uint8_t * __restrict__ result, const int& kbx, const int& kqs) {
    preloaded_data_iq4_xs_q8_1 * preloaded_data = (preloaded_data_iq4_xs_q8_1 *) result;
    const block_iq4_xs * bq4 = (const block_iq4_xs *) vx + kbx;

#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int aux_q4 = get_int_b4(bq4->qs, kqs + j);
        preloaded_data->v_table[j] = get_int_from_table_16(aux_q4, kvalues_iq4nl);
    }

    preloaded_data->ls = ((bq4->scales_l[kqs/8] >> (kqs & 0x04)) & 0x0F) | (((bq4->scales_h >> (kqs/2)) & 0x03) << 4);
    preloaded_data->d_iq4_xs = bq4->d;
}
