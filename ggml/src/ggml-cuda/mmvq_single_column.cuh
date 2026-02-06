#pragma once

#include "mmvq.cuh"
#include "mmvq_preload_data.cuh"
#include "quantize.cuh"
#include "unary.cuh"
#include "vecdotq_preloaded_data.cuh"

#define MMVQ_CEIL(a,b) ((a + b - 1) / b)

// TODO Test on some NVidia GPUs, see if this kernel must be limited to AMD hardware only.
typedef float (*vec_dot_q_cuda_preloaded_t)(const void * __restrict__ preloaded_data_void);

static constexpr __device__ vec_dot_q_cuda_preloaded_t get_vec_dot_q_cuda_preloaded(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:    return vec_dot_q4_0_q8_1_preloaded_data;
        case GGML_TYPE_Q4_1:    return vec_dot_q4_1_q8_1_preloaded_data;
        case GGML_TYPE_Q5_0:    return vec_dot_q5_0_q8_1_preloaded_data;
        case GGML_TYPE_Q5_1:    return vec_dot_q5_1_q8_1_preloaded_data;
        case GGML_TYPE_Q8_0:    return vec_dot_q8_0_q8_1_preloaded_data;
        case GGML_TYPE_MXFP4:   return vec_dot_mxfp4_q8_1_preloaded_data;
        case GGML_TYPE_Q2_K:    return vec_dot_q2_K_q8_1_preloaded_data;
        case GGML_TYPE_Q3_K:    return vec_dot_q3_K_q8_1_preloaded_data;
        case GGML_TYPE_Q4_K:    return vec_dot_q4_K_q8_1_preloaded_data;
        case GGML_TYPE_Q5_K:    return vec_dot_q5_K_q8_1_preloaded_data;
        case GGML_TYPE_Q6_K:    return vec_dot_q6_K_q8_1_preloaded_data;
        case GGML_TYPE_IQ2_XXS: return vec_dot_iq2_xxs_q8_1_preloaded_data;
        case GGML_TYPE_IQ2_XS:  return vec_dot_iq2_xs_q8_1_preloaded_data;
        case GGML_TYPE_IQ2_S:   return vec_dot_iq2_s_q8_1_preloaded_data;
        case GGML_TYPE_IQ3_XXS: return vec_dot_iq3_xxs_q8_1_preloaded_data;
        case GGML_TYPE_IQ3_S:   return vec_dot_iq3_s_q8_1_preloaded_data;
        case GGML_TYPE_IQ1_S:   return vec_dot_iq1_s_q8_1_preloaded_data;
        case GGML_TYPE_IQ1_M:   return vec_dot_iq1_m_q8_1_preloaded_data;
        case GGML_TYPE_IQ4_NL:  return vec_dot_iq4_nl_q8_1_preloaded_data;
        case GGML_TYPE_IQ4_XS:  return vec_dot_iq4_xs_q8_1_preloaded_data;
        default:                return nullptr;
    }
}

typedef void (*x_preloader_t)(const void * __restrict__ vx, uint8_t * __restrict__ result, const int& kbx, const int& kqs);

static constexpr __device__ x_preloader_t get_x_preloader(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:    return x_q4_0_preloader;
        case GGML_TYPE_Q4_1:    return x_q4_1_preloader;
        case GGML_TYPE_Q5_0:    return x_q5_0_preloader;
        case GGML_TYPE_Q5_1:    return x_q5_1_preloader;
        case GGML_TYPE_Q8_0:    return x_q8_0_preloader;
        case GGML_TYPE_MXFP4:   return x_mxfp4_preloader;
        case GGML_TYPE_Q2_K:    return x_q2_K_preloader;
        case GGML_TYPE_Q3_K:    return x_q3_K_preloader;
        case GGML_TYPE_Q4_K:    return x_q4_K_preloader;
        case GGML_TYPE_Q5_K:    return x_q5_K_preloader;
        case GGML_TYPE_Q6_K:    return x_q6_K_preloader;
        case GGML_TYPE_IQ2_XXS: return x_iq2_xxs_preloader;
        case GGML_TYPE_IQ2_XS:  return x_iq2_xs_preloader;
        case GGML_TYPE_IQ2_S:   return x_iq2_s_preloader;
        case GGML_TYPE_IQ3_XXS: return x_iq3_xxs_preloader;
        case GGML_TYPE_IQ3_S:   return x_iq3_s_preloader;
        case GGML_TYPE_IQ1_S:   return x_iq1_s_preloader;
        case GGML_TYPE_IQ1_M:   return x_iq1_m_preloader;
        case GGML_TYPE_IQ4_NL:  return x_iq4_nl_preloader;
        case GGML_TYPE_IQ4_XS:  return x_iq4_xs_preloader;
        default:                return nullptr;
    }
}

typedef void (*y_preloader_t)(const block_q8_1 * __restrict__ y, uint8_t * __restrict__ result, const int& kby, const int& kqs);

static constexpr __device__ y_preloader_t get_y_preloader(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:    return y_q4_0_preloader;
        case GGML_TYPE_Q4_1:    return y_q4_1_preloader;
        case GGML_TYPE_Q5_0:    return y_q5_0_preloader;
        case GGML_TYPE_Q5_1:    return y_q5_1_preloader;
        case GGML_TYPE_Q8_0:    return y_q8_0_preloader;
        case GGML_TYPE_MXFP4:   return y_mxfp4_preloader;
        case GGML_TYPE_Q2_K:    return y_q2_K_preloader;
        case GGML_TYPE_Q3_K:    return y_q3_K_preloader;
        case GGML_TYPE_Q4_K:    return y_q4_K_preloader;
        case GGML_TYPE_Q5_K:    return y_q5_K_preloader;
        case GGML_TYPE_Q6_K:    return y_q6_K_preloader;
        case GGML_TYPE_IQ2_XXS: return y_iq2_xxs_preloader;
        case GGML_TYPE_IQ2_XS:  return y_iq2_xs_preloader;
        case GGML_TYPE_IQ2_S:   return y_iq2_s_preloader;
        case GGML_TYPE_IQ3_XXS: return y_iq3_xxs_preloader;
        case GGML_TYPE_IQ3_S:   return y_iq3_s_preloader;
        case GGML_TYPE_IQ1_S:   return y_iq1_s_preloader;
        case GGML_TYPE_IQ1_M:   return y_iq1_m_preloader;
        case GGML_TYPE_IQ4_NL:  return y_iq4_nl_preloader;
        case GGML_TYPE_IQ4_XS:  return y_iq4_xs_preloader;
        default:                return nullptr;
    }
}

static constexpr __host__ __device__ int calc_nwarps_single_column(int ncols_dst, mmvq_parameter_table_id table_id) {
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
    } else if (table_id == MMVQ_PARAMETERS_RDNA2_PLUS) {
        switch (ncols_dst) {
            case 1:
                return 8;
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

constexpr __host__ __device__ int calc_rows_per_block_single_column(int ncols_dst, int table_id) {
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
    } else if (table_id == MMVQ_PARAMETERS_RDNA2_PLUS) {
        switch (ncols_dst) {
            case 1:
                return 1;
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

std::pair<dim3, dim3> calc_launch_params_single_column(
    const int ncols_dst, const int nrows_x, const int nchannels_dst, const int nsamples_or_ntokens,
    const int warp_size, const mmvq_parameter_table_id table_id) {
    const int64_t nblocks = MMVQ_CEIL(nrows_x, calc_rows_per_block_single_column(ncols_dst, table_id)) * ncols_dst;
    const dim3 block_nums(nblocks, nchannels_dst, nsamples_or_ntokens);
    const dim3 block_dims(warp_size, calc_nwarps_single_column(ncols_dst, table_id), 1);
    return {block_nums, block_dims};
}

// TODO Decide if it is worth to support cases "ncols_dst" != 1.
template <ggml_type type, int ncols_dst, bool has_fusion, bool is_multi_token_id = false>
__launch_bounds__(calc_nwarps_single_column(ncols_dst, get_device_table_id())*ggml_cuda_get_physical_warp_size(), 1)
__global__ void mul_mat_vec_q_single_column(
        const void * __restrict__ vx, const void * __restrict__ vy, const int32_t * __restrict__ ids, const ggml_cuda_mm_fusion_args_device fusion, float * __restrict__ dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const uint32_t ids_stride) {

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr mmvq_parameter_table_id table_id = get_device_table_id();
    constexpr int nwarps = calc_nwarps_single_column(ncols_dst, table_id);
    constexpr int rows_per_cuda_block = calc_rows_per_block_single_column(ncols_dst, table_id);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int preloaded_data_size = get_preloaded_data_size(type);
    constexpr x_preloader_t x_preloader = get_x_preloader(type);
    constexpr y_preloader_t y_preloader = get_y_preloader(type);

    static_assert(rows_per_cuda_block == 1, "More rows may cause an out-of-bounds access.");

    constexpr vec_dot_q_cuda_preloaded_t vec_dot_q_cuda_preloaded = get_vec_dot_q_cuda_preloaded(type);

    const     int col_j = blockIdx.x % ncols_dst;
    const     int tid = warp_size*threadIdx.y + threadIdx.x;
    const     int row0 = rows_per_cuda_block*(blockIdx.x / ncols_dst);
    const     int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * nwarps*warp_size / qi;

    const uint32_t channel_dst = blockIdx.y;

    uint32_t token_idx = 0;
    uint32_t channel_x;
    uint32_t channel_y;
    uint32_t sample_dst;

    if constexpr (is_multi_token_id) {
        // Multi-token MUL_MAT_ID path, adding these in the normal path causes a perf regression for n_tokens=1 case
        token_idx  = blockIdx.z;
        channel_x  = ids[channel_dst + token_idx * ids_stride];
        channel_y  = fastmodulo(channel_dst, nchannels_y);
        sample_dst = 0;
    } else {
        channel_x  = ncols_dst == 1 && ids ? ids[channel_dst]                     : fastdiv(channel_dst, channel_ratio);
        channel_y  = ncols_dst == 1 && ids ? fastmodulo(channel_dst, nchannels_y) : channel_dst;
        sample_dst = blockIdx.z;
    }

    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y    = sample_dst;

    bool use_gate = false;
    bool use_bias = false;
    bool use_gate_bias = false;
    const void * vgate = nullptr;
    const float * x_bias = nullptr;
    const float * gate_bias = nullptr;
    ggml_glu_op active_glu;

    if constexpr (has_fusion) {
        use_gate      = fusion.gate      != nullptr;
        use_bias      = fusion.x_bias    != nullptr;
        use_gate_bias = fusion.gate_bias != nullptr && use_gate;
        vgate         = fusion.gate;
        x_bias        = (const float *) fusion.x_bias;
        gate_bias     = (const float *) fusion.gate_bias;
        active_glu    = fusion.glu_op;
    }

    float x_biases    = { 0.0f };
    float gate_biases = { 0.0f };
    if constexpr (has_fusion) {
        const uint32_t channel_bias = ids ? channel_x : channel_dst;
        if (use_bias) {
            x_bias = x_bias + sample_dst*stride_sample_dst + channel_bias*stride_channel_dst + row0;
            // 1. Hide latency by prefetching bias and gate here
            // 2. load only on threads that won't die after partial sum calculation
            if (threadIdx.x < rows_per_cuda_block && threadIdx.y == 0 &&
                (rows_per_cuda_block == 1 || uint32_t(row0 + threadIdx.x) < stride_col_dst)) {
                x_biases = x_bias[col_j * stride_col_dst + threadIdx.x];
            }
        }
        if (use_gate_bias) {
            gate_bias = gate_bias + sample_dst*stride_sample_dst + channel_bias*stride_channel_dst + row0;
            if (threadIdx.x < rows_per_cuda_block && threadIdx.y == 0 &&
                (rows_per_cuda_block == 1 || uint32_t(row0 + threadIdx.x) < stride_col_dst)) {
                gate_biases = gate_bias[col_j * stride_col_dst + threadIdx.x];
            }
        }
    }

    // partial sum for each thread
    float tmp[rows_per_cuda_block] = {{0.0f}};
    float tmp_gate[rows_per_cuda_block] = {{0.0f}};

    const block_q8_1 * y = ((const block_q8_1 *) vy) + sample_y*stride_sample_y + channel_y*stride_channel_y + col_j*stride_col_y;
    if constexpr (is_multi_token_id) {
        y += token_idx*stride_col_y;
    }
    const int kbx_offset = sample_x*stride_sample_x + channel_x*stride_channel_x + row0*stride_row_x;

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx

        // x block quant index when casting the quants to int
        const int kqs = vdr * (tid % (qi/vdr));

        uint8_t preloaded_data[rows_per_cuda_block][preloaded_data_size];
        uint8_t preloaded_data_gate[rows_per_cuda_block][preloaded_data_size];

        // Preload first block.
        x_preloader(vx, preloaded_data[0], kbx_offset + kbx, kqs);
        y_preloader(y, preloaded_data[0], kby, kqs);
        if constexpr (has_fusion) {
            if (use_gate) {
                x_preloader(vgate, preloaded_data_gate[0], kbx_offset + kbx, kqs);
                y_preloader(y, preloaded_data_gate[0], kby, kqs);
            }
        }

#pragma unroll
        for (int i = 1; i < rows_per_cuda_block; ++i) {
            // Preload block #i.
            x_preloader(vx, preloaded_data[i], kbx_offset + i*stride_row_x + kbx, kqs);
            y_preloader(y, preloaded_data[i], kby, kqs);
            if constexpr (has_fusion) {
                if (use_gate) {
                    x_preloader(vgate, preloaded_data_gate[i], kbx_offset + i*stride_row_x + kbx, kqs);
                    y_preloader(y, preloaded_data_gate[i], kby, kqs);
                }
            }

            // Compute block #{i-1}.
            tmp[i-1] += vec_dot_q_cuda_preloaded(preloaded_data[i-1]);

            if constexpr (has_fusion) {
                if (use_gate) {
                    tmp_gate[i-1] += vec_dot_q_cuda_preloaded(preloaded_data_gate[i-1]);
                }
            }
        }

        // Compute last block.
        tmp[rows_per_cuda_block-1] += vec_dot_q_cuda_preloaded(preloaded_data[rows_per_cuda_block-1]);
        if constexpr (has_fusion) {
            if (use_gate) {
                tmp_gate[rows_per_cuda_block-1] += vec_dot_q_cuda_preloaded(preloaded_data_gate[rows_per_cuda_block-1]);
            }
        }
    }

    __shared__ float tmp_shared[nwarps-1 > 0 ? nwarps-1 : 1][rows_per_cuda_block][warp_size];
    __shared__ float tmp_shared_gate[(has_fusion && (nwarps-1 > 0)) ? nwarps-1 : 1][rows_per_cuda_block][warp_size];
    if constexpr (!has_fusion) {
        (void) tmp_shared_gate;
    } else if (!use_gate) {
        (void) tmp_shared_gate;
    }

    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
            tmp_shared[threadIdx.y-1][i][threadIdx.x] = tmp[i];
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmp_shared_gate[threadIdx.y-1][i][threadIdx.x] = tmp_gate[i];
                }
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

    dst += sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row0;

    if constexpr (is_multi_token_id) {
        dst += token_idx*stride_col_dst;
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
        for (int l = 0; l < nwarps-1; ++l) {
            tmp[i] += tmp_shared[l][i][threadIdx.x];
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmp_gate[i] += tmp_shared_gate[l][i][threadIdx.x];
                }
            }
        }
        tmp[i] = warp_reduce_sum<warp_size>(tmp[i]);
        if constexpr (has_fusion) {
            if (use_gate) {
                tmp_gate[i] = warp_reduce_sum<warp_size>(tmp_gate[i]);
            }
        }
    }

    if (threadIdx.x < rows_per_cuda_block && (rows_per_cuda_block == 1 || uint32_t(row0 + threadIdx.x) < stride_col_dst)) {
        float result = tmp[threadIdx.x];
        if constexpr (has_fusion) {
            if (use_bias) {
                result += x_biases;
            }
            if (use_gate) {
                float gate_value = tmp_gate[threadIdx.x];
                if (use_gate_bias) {
                    gate_value += gate_biases;
                }
                switch (active_glu) {
                    case GGML_GLU_OP_SWIGLU:
                        result *= ggml_cuda_op_silu_single(gate_value);
                        break;
                    case GGML_GLU_OP_GEGLU:
                        result *= ggml_cuda_op_gelu_single(gate_value);
                        break;
                    case GGML_GLU_OP_SWIGLU_OAI: {
                        result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                        break;
                    }
                    default:
                        result = result * gate_value;
                        break;
                }
            }
        }
        dst[col_j*stride_col_dst + threadIdx.x] = result;
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, use_bias, use_gate_bias, active_glu, gate_bias, x_bias, tmp_gate);
    }
}

#undef MMVQ_CEIL
