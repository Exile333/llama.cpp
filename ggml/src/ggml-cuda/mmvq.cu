#include "mmvq.cuh"
#include "quantize.cuh"
#include "vecdotq.cuh"

#include <hip/hip_cooperative_groups.h>

#include <cstdint>

#define CEIL(a,b) ((a + b - 1) / b)

typedef float (*vec_dot_q_cuda_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs);

static constexpr __device__ vec_dot_q_cuda_t get_vec_dot_q_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:    return vec_dot_q4_0_q8_1;
        case GGML_TYPE_Q4_1:    return vec_dot_q4_1_q8_1;
        case GGML_TYPE_Q5_0:    return vec_dot_q5_0_q8_1;
        case GGML_TYPE_Q5_1:    return vec_dot_q5_1_q8_1;
        case GGML_TYPE_Q8_0:    return vec_dot_q8_0_q8_1;
        case GGML_TYPE_MXFP4:   return vec_dot_mxfp4_q8_1;
        case GGML_TYPE_Q2_K:    return vec_dot_q2_K_q8_1;
        case GGML_TYPE_Q3_K:    return vec_dot_q3_K_q8_1;
        case GGML_TYPE_Q4_K:    return vec_dot_q4_K_q8_1;
        case GGML_TYPE_Q5_K:    return vec_dot_q5_K_q8_1;
        case GGML_TYPE_Q6_K:    return vec_dot_q6_K_q8_1;
        case GGML_TYPE_IQ2_XXS: return vec_dot_iq2_xxs_q8_1;
        case GGML_TYPE_IQ2_XS:  return vec_dot_iq2_xs_q8_1;
        case GGML_TYPE_IQ2_S:   return vec_dot_iq2_s_q8_1;
        case GGML_TYPE_IQ3_XXS: return vec_dot_iq3_xxs_q8_1;
        case GGML_TYPE_IQ1_S:   return vec_dot_iq1_s_q8_1;
        case GGML_TYPE_IQ1_M:   return vec_dot_iq1_m_q8_1;
        case GGML_TYPE_IQ4_NL:  return vec_dot_iq4_nl_q8_1;
        case GGML_TYPE_IQ4_XS:  return vec_dot_iq4_xs_q8_1;
        case GGML_TYPE_IQ3_S:   return vec_dot_iq3_s_q8_1;
        default:                return nullptr;
    }
}

typedef float (*vec_dot_q_cuda_preloaded_t)(const void * __restrict__ preloaded_data_void);

static constexpr __device__ vec_dot_q_cuda_preloaded_t get_vec_dot_q_cuda_preloaded(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:    return vec_dot_q4_0_q8_1_preloaded_data;
        case GGML_TYPE_Q4_K:    return nullptr;
        case GGML_TYPE_Q6_K:    return vec_dot_q6_K_q8_1_preloaded_data;
        default:                return nullptr;
    }
}

static constexpr __device__ int get_vdr_mmvq(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:    return VDR_Q4_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_1:    return VDR_Q4_1_Q8_1_MMVQ;
        case GGML_TYPE_Q5_0:    return VDR_Q5_0_Q8_1_MMVQ;
        case GGML_TYPE_Q5_1:    return VDR_Q5_1_Q8_1_MMVQ;
        case GGML_TYPE_Q8_0:    return VDR_Q8_0_Q8_1_MMVQ;
        case GGML_TYPE_MXFP4:   return VDR_MXFP4_Q8_1_MMVQ;
        case GGML_TYPE_Q2_K:    return VDR_Q2_K_Q8_1_MMVQ;
        case GGML_TYPE_Q3_K:    return VDR_Q3_K_Q8_1_MMVQ;
        case GGML_TYPE_Q4_K:    return VDR_Q4_K_Q8_1_MMVQ;
        case GGML_TYPE_Q5_K:    return VDR_Q5_K_Q8_1_MMVQ;
        case GGML_TYPE_Q6_K:    return VDR_Q6_K_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XXS: return VDR_IQ2_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XS:  return VDR_IQ2_XS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_S:   return VDR_IQ2_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_XXS: return VDR_IQ3_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_S:   return VDR_IQ3_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_NL:  return VDR_IQ4_NL_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_XS:  return VDR_IQ4_XS_Q8_1_MMVQ;
        default:                return 1;
    }
}

////////// Q8_1 preload stuff.

static __device__ __forceinline__ int get_q4_k_q8_1_block_offset(const int& tid) {
    // kqs is in 0,2..30. bq8_offset = iqs/4 -> bq8_offset = 0, 2, 4, 6.
    constexpr const int qi  = ggml_cuda_type_traits<GGML_TYPE_Q4_K>::qi;
    constexpr const int vdr = get_vdr_mmvq(GGML_TYPE_Q4_K);
    const int kqs = vdr * (tid % (qi/vdr));

    return QR4_K * ((kqs/2) / (QI8_1/2));
}

typedef void (*y_preloader_t)(const block_q8_1 * __restrict__ y, uint8_t * __restrict__ result, const int& kby, const int& kqs);

static __device__ __forceinline__ void y_q4_0_preloader(const block_q8_1 * __restrict__ y, uint8_t * __restrict__ result, const int& kby, const int& kqs) {
    preloaded_data_q4_0_q8_1 * preloaded_data = (preloaded_data_q4_0_q8_1 *) result;

#pragma unroll
    for (int i = 0; i < VDR_Q4_0_Q8_1_MMVQ; ++i) {
        preloaded_data->scales_q8_1[2*i+0] = get_int_b4(y[kby].qs, kqs + i);
        preloaded_data->scales_q8_1[2*i+1] = get_int_b4(y[kby].qs, kqs + i + QI4_0);
    }
    preloaded_data->ds_q8_1 = y[kby].ds;
}

static __device__ __forceinline__ void y_q6_k_preloader(const block_q8_1 * __restrict__ y, uint8_t * __restrict__ result, const int& kby, const int& kqs) {
    preloaded_data_q6_K_q8_1 * preloaded_data = (preloaded_data_q6_K_q8_1 *) result;
    const int y_offset = 2 * QR6_K * (kqs / (QI6_K/2)) + (kqs % (QI6_K/2)) / (QI6_K/4);

#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        preloaded_data->scales_q8_1[i] = get_int_b4(y[kby + y_offset + 2*i].qs, kqs % QI8_1);
        preloaded_data->ds_q8_1[i] = __low2float(y[kby + y_offset + 2*i].ds);
    }
}

// TODO more types
static constexpr __device__ y_preloader_t get_y_preloader(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:    return y_q4_0_preloader;
        case GGML_TYPE_Q4_K:    return nullptr;
        case GGML_TYPE_Q6_K:    return y_q6_k_preloader;
        default:                return nullptr;
    }
}

///////////////////// x preload stuff

// TODO Infer these params.
// TODO Provide more datatypes.
static constexpr __device__ int get_preloaded_data_size(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:    return sizeof(preloaded_data_q4_0_q8_1);
        case GGML_TYPE_Q4_K:    return 0;
        case GGML_TYPE_Q6_K:    return sizeof(preloaded_data_q6_K_q8_1);
        default:                return 0;
    }
}

typedef void (*x_preloader_t)(const void * __restrict__ vx, uint8_t * __restrict__ result, const int& kbx, const int& kqs);

static __device__ __forceinline__ void x_q4_0_preloader(const void * __restrict__ vx, uint8_t * __restrict__ result, const int& kbx, const int& kqs) {
    preloaded_data_q4_0_q8_1 * preloaded_data = (preloaded_data_q4_0_q8_1 *) result;
    const block_q4_0 * bq4_0 = (const block_q4_0 *) vx + kbx;

#pragma unroll
    for (int i = 0; i < VDR_Q4_0_Q8_1_MMVQ; ++i) {
        preloaded_data->scales_q4_0[i] = get_int_b2(bq4_0->qs, kqs + i);
    }
    preloaded_data->d_q4_0 = bq4_0->d;
}

static __device__ __forceinline__ void x_q4_k_preloader(const void * __restrict__ vx, uint8_t * __restrict__ result, const int& kbx, const int& kqs) {
    ((block_q4_K *) result)[0] = ((const block_q4_K *) vx)[kbx];
}

static __device__ __forceinline__ void x_q6_k_preloader(const void * __restrict__ vx, uint8_t * __restrict__ result, const int& kbx, const int& kqs) {
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

// TODO more types
static constexpr __device__ x_preloader_t get_x_preloader(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:    return x_q4_0_preloader;
        case GGML_TYPE_Q4_K:    return nullptr;
        case GGML_TYPE_Q6_K:    return x_q6_k_preloader;
        default:                return nullptr;
    }
}

enum mmvq_parameter_table_id {
    MMVQ_PARAMETERS_GENERIC = 0,
    MMVQ_PARAMETERS_GCN,
    MMVQ_PARAMETERS_RDNA2
};

static constexpr __device__ mmvq_parameter_table_id get_device_table_id() {
#if defined(RDNA2) || defined(RDNA3) || defined(RDNA4)
    return MMVQ_PARAMETERS_RDNA2;
#elif defined(GCN) || defined(CDNA)
    return MMVQ_PARAMETERS_GCN;
#else
    return MMVQ_PARAMETERS_GENERIC;
#endif
}

static __host__ mmvq_parameter_table_id get_device_table_id(int cc) {
    if (GGML_CUDA_CC_IS_RDNA2(cc) || GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4(cc)) {
        return MMVQ_PARAMETERS_RDNA2;
    }
    if (GGML_CUDA_CC_IS_GCN(cc) || GGML_CUDA_CC_IS_CDNA(cc)) {
        return MMVQ_PARAMETERS_GCN;
    }
    return MMVQ_PARAMETERS_GENERIC;
}

static constexpr __host__ __device__ int calc_nwarps(int ncols_dst,  mmvq_parameter_table_id table_id) {
    if (table_id == MMVQ_PARAMETERS_GENERIC) {
        switch (ncols_dst) {
            case 1:
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    } else if (table_id == MMVQ_PARAMETERS_GCN) {
        switch (ncols_dst) {
            case 1:
            case 2:
            case 3:
            case 4:
                return 2;
            case 5:
            case 6:
            case 7:
            case 8:
            default:
                return 1;
        }
    } else {
        switch (ncols_dst) {
            case 1:
                return 4;
            case 2:
                return 4;
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 4;
            default:
                return 1;
        }
    }
    return 1;
}

static constexpr __host__ __device__ int calc_rows_per_block(int ncols_dst, int table_id) {
    if (table_id == MMVQ_PARAMETERS_GENERIC || table_id == MMVQ_PARAMETERS_GCN) {
        switch (ncols_dst) {
            case 1:
                return 1;
            case 2:
            case 3:
            case 4:
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    } else {
        switch (ncols_dst) {
            case 1:
                return 4;
            case 2:
                return 8;
            case 3:
            case 4:
                return 8;
            case 5:
            case 6:
            case 7:
            case 8:
                return 8;
            default:
                return 1;
        }
    }
    return 1;
}

/*
struct __device__ TMMVQIterState {
    int block_k;
    int row_i;
    int group_start_block_k;
    int group_start_row_i;
    const int blocks_cnt;
    const int rows_cnt;
};

static __device__ __forceinline__ bool mmvq_iterate(int& block_k, int& row_i, const int& rows_cnt, const int& blocks_cnt, const int& blocks_per_iter) {
}
*/

template <ggml_type type, int ncols_dst>
// tell the compiler to use as many registers as it wants, see nwarps definition below
__launch_bounds__(calc_nwarps(ncols_dst, get_device_table_id())*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q(
        const void * __restrict__ vx, const void * __restrict__ vy, const int32_t * __restrict__ ids, float * __restrict__ dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst) {

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qr  = ggml_cuda_type_traits<type>::qr;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr int preloaded_data_size = get_preloaded_data_size(type);
    constexpr x_preloader_t x_preloader = get_x_preloader(type);
    constexpr y_preloader_t y_preloader = get_y_preloader(type);
    constexpr mmvq_parameter_table_id table_id = get_device_table_id();
    constexpr int nwarps = calc_nwarps(ncols_dst, table_id);
    constexpr int rows_per_cuda_block = calc_rows_per_block(ncols_dst, table_id);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    //constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);
    constexpr vec_dot_q_cuda_preloaded_t vec_dot_q_cuda_preloaded = get_vec_dot_q_cuda_preloaded(type);

    const     int col_j = blockIdx.x % ncols_dst;
    const     int tid = warp_size*threadIdx.y + threadIdx.x;
    const     int row0 = rows_per_cuda_block*(blockIdx.x / ncols_dst);
    const     int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * nwarps*warp_size / qi;

    // The MUL_MAT_ID code path with ids != nullptr is only implemented for ncols_dst == 1.
    const uint32_t channel_dst = blockIdx.y;
    const uint32_t channel_x   = ncols_dst == 1 && ids ? ids[channel_dst]                     : fastdiv(channel_dst, channel_ratio);
    const uint32_t channel_y   = ncols_dst == 1 && ids ? fastmodulo(channel_dst, nchannels_y) : channel_dst;
    const uint32_t sample_dst  = blockIdx.z;
    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y    = sample_dst;

    const block_q8_1 * y = ((const block_q8_1 *) vy) + sample_y*stride_sample_y + channel_y*stride_channel_y + col_j*stride_col_y;
    const int kbx_offset = sample_x*stride_sample_x + channel_x*stride_channel_x + row0*stride_row_x;

    float tmp_local[rows_per_cuda_block] = {0.0f};
    __shared__ float tmp_shared[rows_per_cuda_block][nwarps];

    // partial sum for each thread
    {
        constexpr const int preloaded_data_blocks_count = 2;
        constexpr const int process_blocks_per_iteration = 1;
        constexpr const int initial_blocks_cnt = preloaded_data_blocks_count - process_blocks_per_iteration;
        const int kqs = vdr * (tid % (qi/vdr));
        uint8_t preloaded_data[preloaded_data_blocks_count][preloaded_data_size];
        uint8_t row_by_block[preloaded_data_blocks_count] = {0};
        int total_blocks_cnt = 0;
        //bool phase_active = kbx < blocks_per_row_x;

        // Initial preload.
        int preload_kbx = tid / (qi/vdr);
        int preload_kby = preload_kbx * (qk/QK8_1);
        int preload_row_i = 0;
        int block_preload_idx = 0;
        if (preload_kbx < blocks_per_row_x) {
#pragma unroll
            for (int block_i = 0; block_i < initial_blocks_cnt; ++block_i) {
                x_preloader(vx, preloaded_data[block_preload_idx], kbx_offset + preload_row_i*stride_row_x + preload_kbx, kqs);
                y_preloader(y, preloaded_data[block_preload_idx], preload_kby, kqs);
                row_by_block[block_preload_idx] = preload_row_i;
                ++preload_row_i;
                ++total_blocks_cnt;
                //block_preload_idx = (block_preload_idx + 1) % preloaded_data_blocks_count;
                ++block_preload_idx;
                if (preload_row_i >= rows_per_cuda_block) {
                    preload_row_i = 0;
                    preload_kbx += blocks_per_iter;
                    preload_kby = preload_kbx * (qk/QK8_1);
                    if (preload_kbx >= blocks_per_row_x) {
                        break;
                    }
                }
            }
        }

        // Preload + process.
        int block_process_idx = 0;
        while (total_blocks_cnt > 0) {
            // Load.
            if (preload_kbx < blocks_per_row_x) {
#pragma unroll
                for (int j = 0; j < process_blocks_per_iteration; ++j) {
                    x_preloader(vx, preloaded_data[block_preload_idx], kbx_offset + preload_row_i*stride_row_x + preload_kbx, kqs);
                    y_preloader(y, preloaded_data[block_preload_idx], preload_kby, kqs);
                    row_by_block[block_preload_idx] = preload_row_i;
                    ++preload_row_i;
                    ++total_blocks_cnt;
                    block_preload_idx = (block_preload_idx + 1) % preloaded_data_blocks_count;
                    if (preload_row_i >= rows_per_cuda_block) {
                        preload_row_i = 0;
                        preload_kbx += blocks_per_iter;
                        preload_kby = preload_kbx * (qk/QK8_1);
                        if (preload_kbx >= blocks_per_row_x) {
                            break;
                        }
                    }
                }
            }

            // Process.
#pragma unroll
            for (int j = 0; j < process_blocks_per_iteration; ++j) {
                if (total_blocks_cnt == 0) {
                    break;
                }
                tmp_local[row_by_block[block_process_idx]] += vec_dot_q_cuda_preloaded(preloaded_data[block_process_idx]);
                block_process_idx = (block_process_idx + 1) % preloaded_data_blocks_count;
                --total_blocks_cnt;
            }
        }

        // Old version.
        /*
        for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
            // Preload Y blocks.
            const int kby = kbx * (qk/QK8_1);
            y_preloader(y, preloaded_data[0], kby, kqs);
#pragma unroll
            for (int data_block_idx = 1; data_block_idx < preloaded_data_blocks_count; ++data_block_idx) {
                //preloaded_data[data_block_idx] = preloaded_data[0];
                y_preloader(y, preloaded_data[data_block_idx], kby, kqs);
            }
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; i += preloaded_data_blocks_count) {
                // Preload X blocks.
#pragma unroll
                for (int data_block_idx = 0; data_block_idx < preloaded_data_blocks_count; ++data_block_idx) {
                    const int current_i = i + data_block_idx;
                    const bool iteration_active = current_i < rows_per_cuda_block;
                    if (iteration_active) {
                        x_preloader(vx, preloaded_data[data_block_idx], kbx_offset + current_i*stride_row_x + kbx, kqs);
                    }
                }
                // Compute preloaded blocks.
#pragma unroll
                for (int data_block_idx = 0; data_block_idx < preloaded_data_blocks_count; ++data_block_idx) {
                    const int current_i = i + data_block_idx;
                    const bool iteration_active = current_i < rows_per_cuda_block;
                    if (iteration_active) {
                        //tmp_local[i] += vec_dot_q_cuda(
                        //    vx, &y[col_j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                        tmp_local[current_i] += vec_dot_q_cuda_preloaded(preloaded_data[data_block_idx]);
                    }
                }
            }
        }
        */
    }

#pragma unroll
    for (int i = 0; i < rows_per_cuda_block; ++i) {
        float warp_dotproduct = warp_reduce_sum<warp_size>(tmp_local[i]);
        if (threadIdx.x == 0) {
            tmp_shared[i][threadIdx.y] = warp_dotproduct;
        }
    }
    __syncthreads();

    if (threadIdx.y > 0) {
        return;
    }

#pragma unroll
    for (int i = 0; i < rows_per_cuda_block; ++i) {
        tmp_local[i] = warp_reduce_sum<warp_size>(threadIdx.x < nwarps ? tmp_shared[i][threadIdx.x] : 0.0f);
    }

    if (threadIdx.x < rows_per_cuda_block && (rows_per_cuda_block == 1 || uint32_t(row0 + threadIdx.x) < stride_col_dst)) {
        auto* dst_channel = dst + sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row0;
        dst_channel[col_j*stride_col_dst + threadIdx.x] = tmp_local[threadIdx.x];
    }
}

static std::pair<dim3, dim3> calc_launch_params(
        const int ncols_dst, const int nrows_x, const int nchannels_y, const int nsamples_y,
        const int warp_size, const mmvq_parameter_table_id table_id) {
    const int64_t nblocks = CEIL(nrows_x, calc_rows_per_block(ncols_dst, table_id)) * ncols_dst;
    const dim3 block_nums(nblocks, nchannels_y, nsamples_y);
    const dim3 block_dims(warp_size, calc_nwarps(ncols_dst, table_id), 1);
    return {block_nums, block_dims};
}

template <ggml_type type>
static void mul_mat_vec_q_switch_ncols_dst(
        const void * vx, const void * vy, const int32_t * ids, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        cudaStream_t stream) {

    GGML_ASSERT(ncols_x % ggml_blck_size(type) == 0);
    GGML_ASSERT(ncols_dst <= MMVQ_MAX_BATCH_SIZE);

    const uint3 nchannels_y_fd   = ids ? init_fastdiv_values(nchannels_y) : make_uint3(0, 0, 0);
    const uint3 channel_ratio_fd = ids ? make_uint3(0, 0, 0)              : init_fastdiv_values(nchannels_dst / nchannels_x);
    const uint3 sample_ratio_fd  = init_fastdiv_values(nsamples_dst  / nsamples_x);

    const int device = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[device].warp_size;
    const mmvq_parameter_table_id table_id = get_device_table_id(ggml_cuda_info().devices[device].cc);

    GGML_ASSERT(!ids || ncols_dst == 1);
    switch (ncols_dst) {
        case 1: {
            constexpr int c_ncols_dst = 1;
            std::pair<dim3, dim3> dims = calc_launch_params(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q<type, c_ncols_dst><<<dims.first, dims.second, 0, stream>>>
                (vx, vy, ids, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst);
        } break;
        case 2: {
            constexpr int c_ncols_dst = 2;
            std::pair<dim3, dim3> dims = calc_launch_params(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q<type, c_ncols_dst><<<dims.first, dims.second, 0, stream>>>
                (vx, vy, ids, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst);
        } break;
        case 3: {
            constexpr int c_ncols_dst = 3;
            std::pair<dim3, dim3> dims = calc_launch_params(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q<type, c_ncols_dst><<<dims.first, dims.second, 0, stream>>>
                (vx, vy, ids, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst);
        } break;
        case 4: {
            constexpr int c_ncols_dst = 4;
            std::pair<dim3, dim3> dims = calc_launch_params(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q<type, c_ncols_dst><<<dims.first, dims.second, 0, stream>>>
                (vx, vy, ids, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst);
        } break;
        case 5: {
            constexpr int c_ncols_dst = 5;
            std::pair<dim3, dim3> dims = calc_launch_params(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q<type, c_ncols_dst><<<dims.first, dims.second, 0, stream>>>
                (vx, vy, ids, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst);
        } break;
        case 6: {
            constexpr int c_ncols_dst = 6;
            std::pair<dim3, dim3> dims = calc_launch_params(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q<type, c_ncols_dst><<<dims.first, dims.second, 0, stream>>>
                (vx, vy, ids, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst);
        } break;
        case 7: {
            constexpr int c_ncols_dst = 7;
            std::pair<dim3, dim3> dims = calc_launch_params(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q<type, c_ncols_dst><<<dims.first, dims.second, 0, stream>>>
                (vx, vy, ids, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst);
        } break;
        case 8: {
            constexpr int c_ncols_dst = 8;
            std::pair<dim3, dim3> dims = calc_launch_params(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q<type, c_ncols_dst><<<dims.first, dims.second, 0, stream>>>
                (vx, vy, ids, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst);
        } break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

static void mul_mat_vec_q_switch_type(
        const void * vx, const ggml_type type_x, const void * vy, const int32_t * ids, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        cudaStream_t stream) {
    switch (type_x) {
        case GGML_TYPE_Q4_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_0>
                (vx, vy, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_1>
                (vx, vy, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_0>
                (vx, vy, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_1>
                (vx, vy, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q8_0>
                (vx, vy, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 stream);
            break;
        case GGML_TYPE_MXFP4:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_MXFP4>
                (vx, vy, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 stream);
            break;
        case GGML_TYPE_Q2_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_K>
                (vx, vy, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q3_K>
                (vx, vy, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_K>
                (vx, vy, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_K>
                (vx, vy, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q6_K>
                (vx, vy, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XXS>
                (vx, vy, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XS>
                (vx, vy, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_S>
                (vx, vy, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_XXS>
                (vx, vy, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 stream);
            break;
        case GGML_TYPE_IQ1_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_S>
                (vx, vy, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 stream);
            break;
        case GGML_TYPE_IQ1_M:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_M>
                (vx, vy, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_NL>
                (vx, vy, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_XS>
                (vx, vy, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_S>
                (vx, vy, ids, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

void ggml_cuda_mul_mat_vec_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    GGML_ASSERT(!ids || ne12 == 1); // Implementation is only correct for batch size 1.

    const float   * src1_d =       (const float   *) src1->data;
    const int32_t *  ids_d = ids ? (const int32_t *)  ids->data : nullptr;
    float         *  dst_d =       (float         *)  dst->data;

    // If src0 is a temporary compute buffer, clear any potential padding.
    if (ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream));
        }
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1);
    {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;
        quantize_row_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
    }

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s11 = ne10_padded / QK8_1;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const int64_t s12 = ne11*s11;
    const int64_t s13 = ne12*s12;

    // For MUL_MAT_ID the memory layout is different than for MUL_MAT:
    const int64_t ncols_dst          = ids ? ne2  : ne1;
    const int64_t nchannels_y        = ids ? ne11 : ne12;
    const int64_t nchannels_dst      = ids ? ne1  : ne2;
    const int64_t stride_col_dst     = ids ? s2   : s1;
    const int64_t stride_col_y       = ids ? s12  : s11;
    const int64_t stride_channel_dst = ids ? s1   : s2;
    const int64_t stride_channel_y   = ids ? s11  : s12;

    mul_mat_vec_q_switch_type(
        src0->data, src0->type, src1_q8_1.get(), ids_d, dst_d, ne00,
        ne01,              ncols_dst,     s01, stride_col_y,     stride_col_dst,
        ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
        ne03,              ne3,           s03, s13,              s3,                 stream);
}

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];
    const int64_t row_diff = row_high - row_low;

    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    int id = ggml_cuda_get_device();

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    const int stride_row_x = ne00 / ggml_blck_size(src0->type);
    const int stride_col_y = src1_padded_row_size / QK8_1;

    mul_mat_vec_q_switch_type(
        src0_dd_i, src0->type, src1_ddq_i, nullptr, dst_dd_i, ne00, row_diff, src1_ncols, stride_row_x, stride_col_y, nrows_dst,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, stream);

    GGML_UNUSED_VARS(src1, dst, src1_ddf_i, src1_ncols, src1_padded_row_size);
}
