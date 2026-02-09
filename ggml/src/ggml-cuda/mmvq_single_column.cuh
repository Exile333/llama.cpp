#pragma once

#include "mmvq.cuh"
#include "quantize.cuh"
#include "unary.cuh"

#define MMVQ_CEIL(a,b) ((a + b - 1) / b)

// TODO Test on some NVidia GPUs, see if this kernel must be limited to AMD hardware only.
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

    static_assert(rows_per_cuda_block == 1, "More rows may cause an out-of-bounds access.");

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

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

#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
            tmp[i] += vec_dot_q_cuda(
                vx, &y[kby], kbx_offset + i*stride_row_x + kbx, kqs);
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmp_gate[i] += vec_dot_q_cuda(
                        vgate, &y[kby], kbx_offset + i*stride_row_x + kbx, kqs);
                }
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
