#pragma once

// ============= Q4_1 =============

template <int vdr> static __device__ __forceinline__ float vec_dot_q4_1_q8_1_impl_preloaded_data(
    const int * v, const int * u, const half2 & dm4, const half2 & ds8) {

    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        const int vi0 = (v[i] >> 0) & 0x0F0F0F0F;
        const int vi1 = (v[i] >> 4) & 0x0F0F0F0F;

        sumi = ggml_cuda_dp4a(vi0, u[2*i+0], sumi);
        sumi = ggml_cuda_dp4a(vi1, u[2*i+1], sumi);
    }

#ifdef FAST_FP16_AVAILABLE
    const float2 tmp = __half22float2(__hmul2(dm4, ds8));
    const float d4d8 = tmp.x;
    const float m4s8 = tmp.y;
#else
    const float2 dm4f = __half22float2(dm4);
    const float2 ds8f = __half22float2(ds8);
    const float d4d8 = dm4f.x * ds8f.x;
    const float m4s8 = dm4f.y * ds8f.y;
#endif

    return sumi * d4d8 + m4s8 / (QI8_1 / (vdr * QR4_1));
}

struct preloaded_data_q4_1_q8_1{
    int scales_q8_1[2*VDR_Q4_1_Q8_1_MMVQ];
    int scales_q4_1[VDR_Q4_1_Q8_1_MMVQ];
    ggml_half2 ds_q8_1;
    ggml_half2 dm_q4_1;
};

static __device__ __forceinline__ float vec_dot_q4_1_q8_1_preloaded_data(const void * __restrict__ preloaded_data_void) {
    const preloaded_data_q4_1_q8_1 * preloaded_data = (const preloaded_data_q4_1_q8_1 *) preloaded_data_void;

    return vec_dot_q4_1_q8_1_impl_preloaded_data<VDR_Q4_1_Q8_1_MMVQ>(
        preloaded_data->scales_q4_1,
        preloaded_data->scales_q8_1,
        preloaded_data->dm_q4_1,
        preloaded_data->ds_q8_1);
}

// ============= Q5_0 =============

template <int vdr> static __device__ __forceinline__ float vec_dot_q5_0_q8_1_impl_preloaded_data(
    const int * vl, const int * vh, const int * u, const float & d5, const half2 & ds8) {

    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        int vi0 = (vl[i] >>  0) & 0x0F0F0F0F;
        vi0    |= (vh[i] <<  4) & 0x00000010;
        vi0    |= (vh[i] << 11) & 0x00001000;
        vi0    |= (vh[i] << 18) & 0x00100000;
        vi0    |= (vh[i] << 25) & 0x10000000;
        sumi = ggml_cuda_dp4a(vi0, u[2*i+0], sumi);

        int vi1 = (vl[i] >>  4) & 0x0F0F0F0F;
        vi1    |= (vh[i] >> 12) & 0x00000010;
        vi1    |= (vh[i] >>  5) & 0x00001000;
        vi1    |= (vh[i] <<  2) & 0x00100000;
        vi1    |= (vh[i] <<  9) & 0x10000000;
        sumi = ggml_cuda_dp4a(vi1, u[2*i+1], sumi);
    }

    const float2 ds8f = __half22float2(ds8);

    return d5 * (sumi * ds8f.x - (16*vdr/QI5_0) * ds8f.y);
}

struct preloaded_data_q5_0_q8_1{
    int vl[VDR_Q5_0_Q8_1_MMVQ];
    int vh[VDR_Q5_0_Q8_1_MMVQ];
    int scales_q8_1[2*VDR_Q5_0_Q8_1_MMVQ];
    ggml_half2 ds_q8_1;
    ggml_half d_q5_0;
};

static __device__ __forceinline__ float vec_dot_q5_0_q8_1_preloaded_data(const void * __restrict__ preloaded_data_void) {
    const preloaded_data_q5_0_q8_1 * preloaded_data = (const preloaded_data_q5_0_q8_1 *) preloaded_data_void;

    return vec_dot_q5_0_q8_1_impl_preloaded_data<VDR_Q5_0_Q8_1_MMVQ>(
        preloaded_data->vl,
        preloaded_data->vh,
        preloaded_data->scales_q8_1,
        preloaded_data->d_q5_0,
        preloaded_data->ds_q8_1);
}

// ============= Q5_1 =============

template <int vdr> static __device__ __forceinline__ float vec_dot_q5_1_q8_1_impl_preloaded_data(
    const int * vl, const int * vh, const int * u, const half2 & dm5, const half2 & ds8) {

    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        int vi0 = (vl[i] >>  0) & 0x0F0F0F0F;
        vi0    |= (vh[i] <<  4) & 0x00000010;
        vi0    |= (vh[i] << 11) & 0x00001000;
        vi0    |= (vh[i] << 18) & 0x00100000;
        vi0    |= (vh[i] << 25) & 0x10000000;
        sumi = ggml_cuda_dp4a(vi0, u[2*i+0], sumi);

        int vi1 = (vl[i] >>  4) & 0x0F0F0F0F;
        vi1    |= (vh[i] >> 12) & 0x00000010;
        vi1    |= (vh[i] >>  5) & 0x00001000;
        vi1    |= (vh[i] <<  2) & 0x00100000;
        vi1    |= (vh[i] <<  9) & 0x10000000;
        sumi = ggml_cuda_dp4a(vi1, u[2*i+1], sumi);
    }

#ifdef FAST_FP16_AVAILABLE
    const float2 tmp = __half22float2(__hmul2(dm5, ds8));
    const float d5d8 = tmp.x;
    const float m5s8 = tmp.y;
#else
    const float2 dm5f = __half22float2(dm5);
    const float2 ds8f = __half22float2(ds8);
    const float d5d8 = dm5f.x * ds8f.x;
    const float m5s8 = dm5f.y * ds8f.y;
#endif

    return sumi*d5d8 + m5s8 / (QI5_1 / vdr);
}

struct preloaded_data_q5_1_q8_1{
    int vl[VDR_Q5_1_Q8_1_MMVQ];
    int vh[VDR_Q5_1_Q8_1_MMVQ];
    int scales_q8_1[2*VDR_Q5_1_Q8_1_MMVQ];
    ggml_half2 ds_q8_1;
    ggml_half2 dm_q5_1;
};

static __device__ __forceinline__ float vec_dot_q5_1_q8_1_preloaded_data(const void * __restrict__ preloaded_data_void) {
    const preloaded_data_q5_1_q8_1 * preloaded_data = (const preloaded_data_q5_1_q8_1 *) preloaded_data_void;

    return vec_dot_q5_1_q8_1_impl_preloaded_data<VDR_Q5_1_Q8_1_MMVQ>(
        preloaded_data->vl,
        preloaded_data->vh,
        preloaded_data->scales_q8_1,
        preloaded_data->dm_q5_1,
        preloaded_data->ds_q8_1);
}

// ============= Q8_0 =============

struct preloaded_data_q8_0_q8_1{
    int scales_q8_0[VDR_Q8_0_Q8_1_MMVQ];
    int scales_q8_1[VDR_Q8_0_Q8_1_MMVQ];
    ggml_half d_q8_0;
    ggml_half d_q8_1;
};

static __device__ __forceinline__ float vec_dot_q8_0_q8_1_preloaded_data(const void * __restrict__ preloaded_data_void) {
    const preloaded_data_q8_0_q8_1 * preloaded_data = (const preloaded_data_q8_0_q8_1 *) preloaded_data_void;

    return vec_dot_q8_0_q8_1_impl<float, VDR_Q8_0_Q8_1_MMVQ>(
        preloaded_data->scales_q8_0,
        preloaded_data->scales_q8_1,
        preloaded_data->d_q8_0,
        preloaded_data->d_q8_1);
}

// ============= MXFP4 =============

struct preloaded_data_mxfp4_q8_1{
    int aux_q4[VDR_MXFP4_Q8_1_MMVQ];
    int u_q8_1[VDR_MXFP4_Q8_1_MMVQ + 4];
    uint8_t e_mxfp4;
    ggml_half2 ds_q8_1;
};

static __device__ __forceinline__ float vec_dot_mxfp4_q8_1_preloaded_data(const void * __restrict__ preloaded_data_void) {
    const preloaded_data_mxfp4_q8_1 * preloaded_data = (const preloaded_data_mxfp4_q8_1 *) preloaded_data_void;

    int sumi = 0;
#pragma unroll
    for (int l = 0; l < VDR_MXFP4_Q8_1_MMVQ; ++l) {
        const int2 v = get_int_from_table_16(preloaded_data->aux_q4[l], kvalues_mxfp4);

        sumi = ggml_cuda_dp4a(v.x, preloaded_data->u_q8_1[l + 0], sumi);
        sumi = ggml_cuda_dp4a(v.y, preloaded_data->u_q8_1[l + 4], sumi);
    }

    const float d = ggml_cuda_e8m0_to_fp32(preloaded_data->e_mxfp4) * 0.5f * __low2float(preloaded_data->ds_q8_1);
    return d * sumi;
}

// ============= Q2_K =============

struct preloaded_data_q2_K_q8_1{
    int v;
    int scales_q8_1[QR2_K];
    float ds_q8_1[QR2_K];
    uint8_t scales_q2_K[2*QR2_K];
    ggml_half2 dm_q2_K;
};

static __device__ __forceinline__ float vec_dot_q2_K_q8_1_preloaded_data(const void * __restrict__ preloaded_data_void) {
    const preloaded_data_q2_K_q8_1 * preloaded_data = (const preloaded_data_q2_K_q8_1 *) preloaded_data_void;

    return vec_dot_q2_K_q8_1_impl_mmvq(
        preloaded_data->v,
        preloaded_data->scales_q8_1,
        preloaded_data->scales_q2_K,
        preloaded_data->dm_q2_K,
        preloaded_data->ds_q8_1);
}

// ============= Q3_K =============

struct preloaded_data_q3_K_q8_1{
    int vl;
    int vh;
    int scales_q8_1[QR3_K];
    float ds_q8_1[QR3_K];
    uint8_t scales_q3_K[(QK_K/32) + (QK_K/64)];
    int scale_offset;
    ggml_half d_q3_K;
};

static __device__ __forceinline__ float vec_dot_q3_K_q8_1_preloaded_data(const void * __restrict__ preloaded_data_void) {
    const preloaded_data_q3_K_q8_1 * preloaded_data = (const preloaded_data_q3_K_q8_1 *) preloaded_data_void;

    return vec_dot_q3_K_q8_1_impl_mmvq(
        preloaded_data->vl,
        preloaded_data->vh,
        preloaded_data->scales_q8_1,
        preloaded_data->scales_q3_K,
        preloaded_data->scale_offset,
        preloaded_data->d_q3_K,
        preloaded_data->ds_q8_1);
}

// ============= Q4_K =============

struct preloaded_data_q4_K_q8_1{
    int v[2];               // Q4_K quantized data
    int u[2*QR4_K];         // Q8_1 quantized data
    uint16_t q4_K_scm[3];   // Scales + mins, in 6 bits.
    bool q4_K_scm_extended; // Check if one needs to use q4_K_scm[2].
    ggml_half2 d8[QR4_K];   // Q8_1 scale factors
    ggml_half2 dm_q4_K;     // Q4_K d and m values
};

static __device__ __forceinline__ float vec_dot_q4_K_q8_1_preloaded_data(const void * __restrict__ preloaded_data_void) {
    const preloaded_data_q4_K_q8_1 * preloaded_data = (const preloaded_data_q4_K_q8_1 *) preloaded_data_void;

    uint16_t aux[2];
    if (!preloaded_data->q4_K_scm_extended) {
        aux[0] = preloaded_data->q4_K_scm[0] & 0x3f3f;
        aux[1] = preloaded_data->q4_K_scm[1] & 0x3f3f;
    } else {
        aux[0] = ((preloaded_data->q4_K_scm[1] >> 0) & 0x0f0f) | ((preloaded_data->q4_K_scm[2] & 0xc0c0) >> 2);
        aux[1] = ((preloaded_data->q4_K_scm[1] >> 4) & 0x0f0f) | ((preloaded_data->q4_K_scm[0] & 0xc0c0) >> 2);
    }
    const uint8_t * sc = (const uint8_t *)aux;
    const uint8_t * m  = sc + 2;

    float d8_f[QR4_K];
#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        d8_f[i] = __low2float(preloaded_data->d8[i]);
    }

    return vec_dot_q4_K_q8_1_impl_vmmq(
        preloaded_data->v,
        preloaded_data->u,
        sc,
        m,
        preloaded_data->dm_q4_K,
        d8_f);
}

// ============= Q5_K =============

struct preloaded_data_q5_K_q8_1{
    int vl[2];
    int vh[2];
    int scales_q8_1[2*QR5_K];
    float ds_q8_1[QR5_K];
    uint8_t sc[QR5_K];
    uint8_t m[QR5_K];
    ggml_half2 dm_q5_K;
};

static __device__ __forceinline__ float vec_dot_q5_K_q8_1_preloaded_data(const void * __restrict__ preloaded_data_void) {
    const preloaded_data_q5_K_q8_1 * preloaded_data = (const preloaded_data_q5_K_q8_1 *) preloaded_data_void;

    return vec_dot_q5_K_q8_1_impl_vmmq(
        preloaded_data->vl,
        preloaded_data->vh,
        preloaded_data->scales_q8_1,
        preloaded_data->sc,
        preloaded_data->m,
        preloaded_data->dm_q5_K,
        preloaded_data->ds_q8_1);
}

// ============= IQ2_XXS =============

struct preloaded_data_iq2_xxs_q8_1{
    int q2;
    uint32_t aux32;
    int scales_q8_1[8];
    ggml_half d_iq2_xxs;
    ggml_half d_q8_1;
};

static __device__ __forceinline__ float vec_dot_iq2_xxs_q8_1_preloaded_data(const void * __restrict__ preloaded_data_void) {
    const preloaded_data_iq2_xxs_q8_1 * preloaded_data = (const preloaded_data_iq2_xxs_q8_1 *) preloaded_data_void;

    const uint8_t * aux8 = (const uint8_t *) &preloaded_data->q2;

    int sumi = 0;
#pragma unroll
    for (int k0 = 0; k0 < 8; k0 += 2) {
        const int * grid_pos = (const int *) (iq2xxs_grid + aux8[k0/2]);
        const int signs_packed = ksigns_iq2xs[(preloaded_data->aux32 >> (7*k0/2)) & 0x7F];

        const int signs0 = __vcmpne4(((signs_packed & 0x03) << 7) | ((signs_packed & 0x0C) << 21), 0x00000000);
        const int grid0 = __vsub4(grid_pos[0] ^ signs0, signs0);
        sumi = ggml_cuda_dp4a(grid0, preloaded_data->scales_q8_1[k0 + 0], sumi);

        const int signs1 = __vcmpne4(((signs_packed & 0x30) << 3) | ((signs_packed & 0xC0) << 17), 0x00000000);
        const int grid1 = __vsub4(grid_pos[1] ^ signs1, signs1);
        sumi = ggml_cuda_dp4a(grid1, preloaded_data->scales_q8_1[k0 + 1], sumi);
    }

    const int ls = preloaded_data->aux32 >> 28;
    sumi = (ls*sumi + sumi/2)/4;
    const float d = __half2float(preloaded_data->d_iq2_xxs) * __half2float(preloaded_data->d_q8_1);
    return d * sumi;
}

// ============= IQ2_XS =============

struct preloaded_data_iq2_xs_q8_1{
    int2 q2_packed;
    int ls0;
    int ls1;
    int scales_q8_1[8];
    ggml_half d_iq2_xs;
    ggml_half d_q8_1;
};

static __device__ __forceinline__ float vec_dot_iq2_xs_q8_1_preloaded_data(const void * __restrict__ preloaded_data_void) {
    const preloaded_data_iq2_xs_q8_1 * preloaded_data = (const preloaded_data_iq2_xs_q8_1 *) preloaded_data_void;

    const uint16_t * q2 = (const uint16_t *) &preloaded_data->q2_packed;

    int sumi0 = 0;
    int sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint32_t * grid_pos = (const uint32_t *)(iq2xs_grid + (q2[l0/2] & 0x000001FF));
        const uint32_t * signs    = (const uint32_t *)(ksigns64   + (q2[l0/2] >> 9));

        const int grid_l = __vsub4(grid_pos[0] ^ signs[0], signs[0]);
        const int grid_h = __vsub4(grid_pos[1] ^ signs[1], signs[1]);

        if (l0 < 4) {
            sumi0 = ggml_cuda_dp4a(grid_l, preloaded_data->scales_q8_1[l0 + 0], sumi0);
            sumi0 = ggml_cuda_dp4a(grid_h, preloaded_data->scales_q8_1[l0 + 1], sumi0);
        } else {
            sumi1 = ggml_cuda_dp4a(grid_l, preloaded_data->scales_q8_1[l0 + 0], sumi1);
            sumi1 = ggml_cuda_dp4a(grid_h, preloaded_data->scales_q8_1[l0 + 1], sumi1);
        }
    }
    const int sumi = (sumi0*preloaded_data->ls0 + sumi1*preloaded_data->ls1 + (sumi0 + sumi1)/2)/4;
    const float d = __half2float(preloaded_data->d_iq2_xs) * __half2float(preloaded_data->d_q8_1);
    return d * sumi;
}

// ============= IQ2_S =============

struct preloaded_data_iq2_s_q8_1{
    int qs_packed;
    int qh;
    int signs_packed_32;
    int ls0;
    int ls1;
    int scales_q8_1[8];
    ggml_half d_iq2_s;
    ggml_half d_q8_1;
};

static __device__ __forceinline__ float vec_dot_iq2_s_q8_1_preloaded_data(const void * __restrict__ preloaded_data_void) {
    const preloaded_data_iq2_s_q8_1 * preloaded_data = (const preloaded_data_iq2_s_q8_1 *) preloaded_data_void;

    const uint8_t * qs = (const uint8_t *) &preloaded_data->qs_packed;
    const uint8_t * signs_packed_8 = (const uint8_t *) &preloaded_data->signs_packed_32;

    int sumi0 = 0;
    int sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int * grid_pos = (const int *)(iq2s_grid + (qs[l0/2] | ((preloaded_data->qh << (8-l0)) & 0x300)));

        const int signs0 = __vcmpne4(((signs_packed_8[l0/2] & 0x03) << 7) | ((signs_packed_8[l0/2] & 0x0C) << 21), 0x00000000);
        const int signs1 = __vcmpne4(((signs_packed_8[l0/2] & 0x30) << 3) | ((signs_packed_8[l0/2] & 0xC0) << 17), 0x00000000);

        const int grid_l = __vsub4(grid_pos[0] ^ signs0, signs0);
        const int grid_h = __vsub4(grid_pos[1] ^ signs1, signs1);

        if (l0 < 4) {
            sumi0 = ggml_cuda_dp4a(grid_l, preloaded_data->scales_q8_1[l0 + 0], sumi0);
            sumi0 = ggml_cuda_dp4a(grid_h, preloaded_data->scales_q8_1[l0 + 1], sumi0);
        } else {
            sumi1 = ggml_cuda_dp4a(grid_l, preloaded_data->scales_q8_1[l0 + 0], sumi1);
            sumi1 = ggml_cuda_dp4a(grid_h, preloaded_data->scales_q8_1[l0 + 1], sumi1);
        }
    }
    const int sumi = (sumi0*preloaded_data->ls0 + sumi1*preloaded_data->ls1 + (sumi0 + sumi1)/2)/4;

    const float d = __half2float(preloaded_data->d_iq2_s) * __half2float(preloaded_data->d_q8_1);
    return d * sumi;
}

// ============= IQ3_XXS =============

struct preloaded_data_iq3_xxs_q8_1{
    int2 q3_packed;
    uint32_t aux32;
    int scales_q8_1[8];
    ggml_half d_iq3_xxs;
    ggml_half d_q8_1;
};

static __device__ __forceinline__ float vec_dot_iq3_xxs_q8_1_preloaded_data(const void * __restrict__ preloaded_data_void) {
    const preloaded_data_iq3_xxs_q8_1 * preloaded_data = (const preloaded_data_iq3_xxs_q8_1 *) preloaded_data_void;

    const uint8_t * q3 = (const uint8_t *) &preloaded_data->q3_packed;

    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(iq3xxs_grid[q3[l0 + 0]], iq3xxs_grid[q3[l0 + 1]]);

        const int * signs = (const int *)(ksigns64 + ((preloaded_data->aux32 >> (7*l0/2)) & 0x7F));

        const int grid_l = __vsub4(grid_pos.x ^ signs[0], signs[0]);
        const int grid_h = __vsub4(grid_pos.y ^ signs[1], signs[1]);

        sumi = ggml_cuda_dp4a(grid_l, preloaded_data->scales_q8_1[l0 + 0], sumi);
        sumi = ggml_cuda_dp4a(grid_h, preloaded_data->scales_q8_1[l0 + 1], sumi);
    }

    const int ls = preloaded_data->aux32 >> 28;
    sumi = (ls*sumi + sumi/2)/2;
    const float d = __half2float(preloaded_data->d_iq3_xxs) * __half2float(preloaded_data->d_q8_1);
    return d * sumi;
}

// ============= IQ3_S =============

struct preloaded_data_iq3_s_q8_1{
    int2 qs_packed;
    int qh;
    int signs_packed_32;
    int scale_val;
    int scales_q8_1[8];
    ggml_half d_iq3_s;
    ggml_half d_q8_1;
};

static __device__ __forceinline__ float vec_dot_iq3_s_q8_1_preloaded_data(const void * __restrict__ preloaded_data_void) {
    const preloaded_data_iq3_s_q8_1 * preloaded_data = (const preloaded_data_iq3_s_q8_1 *) preloaded_data_void;

    const uint8_t * qs = (const uint8_t *) &preloaded_data->qs_packed;
    const uint8_t * signs_packed_8 = (const uint8_t *) &preloaded_data->signs_packed_32;

    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(
            iq3s_grid[qs[l0 + 0] | ((preloaded_data->qh << (8 - l0)) & 0x100)],
            iq3s_grid[qs[l0 + 1] | ((preloaded_data->qh << (7 - l0)) & 0x100)]);

        const int signs0 = __vcmpne4(((signs_packed_8[l0/2] & 0x03) << 7) | ((signs_packed_8[l0/2] & 0x0C) << 21), 0x00000000);
        const int signs1 = __vcmpne4(((signs_packed_8[l0/2] & 0x30) << 3) | ((signs_packed_8[l0/2] & 0xC0) << 17), 0x00000000);

        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);

        sumi = ggml_cuda_dp4a(grid_l, preloaded_data->scales_q8_1[l0 + 0], sumi);
        sumi = ggml_cuda_dp4a(grid_h, preloaded_data->scales_q8_1[l0 + 1], sumi);
    }

    sumi *= preloaded_data->scale_val;

    const float d = __half2float(preloaded_data->d_iq3_s) * __half2float(preloaded_data->d_q8_1);
    return d * sumi;
}

// ============= IQ1_S =============

struct preloaded_data_iq1_s_q8_1{
    int qs_packed;
    int qh;
    int scales_q8_1[8];
    float d1q;
    float delta;
    ggml_half2 ds_q8_1;
};

static __device__ __forceinline__ float vec_dot_iq1_s_q8_1_preloaded_data(const void * __restrict__ preloaded_data_void) {
    const preloaded_data_iq1_s_q8_1 * preloaded_data = (const preloaded_data_iq1_s_q8_1 *) preloaded_data_void;

    const uint8_t * qs = (const uint8_t *) &preloaded_data->qs_packed;

    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int grid = iq1s_grid_gpu[qs[l0/2] | (((preloaded_data->qh >> 3*(l0/2)) & 0x07) << 8)];

        const int grid0 = (grid >> 0) & 0x0F0F0F0F;
        const int grid1 = (grid >> 4) & 0x0F0F0F0F;

        sumi = ggml_cuda_dp4a(grid0, preloaded_data->scales_q8_1[l0 + 0], sumi);
        sumi = ggml_cuda_dp4a(grid1, preloaded_data->scales_q8_1[l0 + 1], sumi);
    }

    const float2 ds = __half22float2(preloaded_data->ds_q8_1);
    return preloaded_data->d1q * (ds.x*sumi + ds.y*preloaded_data->delta);
}

// ============= IQ1_M =============

struct preloaded_data_iq1_m_q8_1{
    int qs_packed;
    int qh[2];
    int scales_q8_1[8];
    float d;
    int sc0;
    int sc1;
};

static __device__ __forceinline__ float vec_dot_iq1_m_q8_1_preloaded_data(const void * __restrict__ preloaded_data_void) {
    const preloaded_data_iq1_m_q8_1 * preloaded_data = (const preloaded_data_iq1_m_q8_1 *) preloaded_data_void;

    const uint8_t * qs = (const uint8_t *) &preloaded_data->qs_packed;

    int sumi[2] = {0};
    float sumf[2] = {0.0f};
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int qhl = preloaded_data->qh[l0/4];

        const int grid = iq1s_grid_gpu[qs[l0/2] | ((qhl & 0x07) << 8)];

        const int grid0 = (grid >> 0) & 0x0F0F0F0F;
        const int grid1 = (grid >> 4) & 0x0F0F0F0F;

        sumi[l0/4] = ggml_cuda_dp4a(grid0, preloaded_data->scales_q8_1[l0 + 0], sumi[l0/4]);
        sumi[l0/4] = ggml_cuda_dp4a(grid1, preloaded_data->scales_q8_1[l0 + 1], sumi[l0/4]);

        const float delta = -1.0f + IQ1M_DELTA - (qhl & 0x08) * (2.0f*IQ1M_DELTA/0x08);
        int sumy = 0;
        sumy = ggml_cuda_dp4a(preloaded_data->scales_q8_1[l0 + 0], 0x01010101, sumy);
        sumy = ggml_cuda_dp4a(preloaded_data->scales_q8_1[l0 + 1], 0x01010101, sumy);
        sumf[l0/4] += delta*sumy;
    }

    return preloaded_data->d * ((sumi[0] + sumf[0]) * preloaded_data->sc0 + (sumi[1] + sumf[1]) * preloaded_data->sc1);
}

// ============= IQ4_NL =============

struct preloaded_data_iq4_nl_q8_1{
    int2 v_table[VDR_IQ4_NL_Q8_1_MMVQ];
    int scales_q8_1[VDR_IQ4_NL_Q8_1_MMVQ + 4];
    ggml_half d_iq4_nl;
    ggml_half d_q8_1;
};

static __device__ __forceinline__ float vec_dot_iq4_nl_q8_1_preloaded_data(const void * __restrict__ preloaded_data_void) {
    const preloaded_data_iq4_nl_q8_1 * preloaded_data = (const preloaded_data_iq4_nl_q8_1 *) preloaded_data_void;

    int sumi = 0;
#pragma unroll
    for (int l = 0; l < VDR_IQ4_NL_Q8_1_MMVQ; ++l) {
        sumi = ggml_cuda_dp4a(preloaded_data->v_table[l].x, preloaded_data->scales_q8_1[l + 0], sumi);
        sumi = ggml_cuda_dp4a(preloaded_data->v_table[l].y, preloaded_data->scales_q8_1[l + 4], sumi);
    }

    const float d = __half2float(preloaded_data->d_iq4_nl) * __half2float(preloaded_data->d_q8_1);
    return d * sumi;
}

// ============= IQ4_XS =============

struct preloaded_data_iq4_xs_q8_1{
    int2 v_table[4];
    int scales_q8_1[8];
    int ls;
    ggml_half d_iq4_xs;
    ggml_half d_q8_1;
};

static __device__ __forceinline__ float vec_dot_iq4_xs_q8_1_preloaded_data(const void * __restrict__ preloaded_data_void) {
    const preloaded_data_iq4_xs_q8_1 * preloaded_data = (const preloaded_data_iq4_xs_q8_1 *) preloaded_data_void;

    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        sumi = ggml_cuda_dp4a(preloaded_data->v_table[j].x, preloaded_data->scales_q8_1[j + 0], sumi);
        sumi = ggml_cuda_dp4a(preloaded_data->v_table[j].y, preloaded_data->scales_q8_1[j + 4], sumi);
    }

    sumi *= preloaded_data->ls - 32;

    const float d = __half2float(preloaded_data->d_iq4_xs) * __half2float(preloaded_data->d_q8_1);
    return d * sumi;
}
