#ifndef ROPE_CUH
#define ROPE_CUH

#include <cuda_runtime.h>
#include "cuda_utils.cuh"

typedef struct {
    float base_frequency;   // RoPE base frequency
    float scaling;          // RoPE scaling factor
    float* rope_freqs;      // array of precomputed RoPE frequencies
} RoPEContext;

/**
 * @brief Applies Rotary Position Embedding (RoPE) with scaling to query and key tensors.
 *
 * @param q Pointer to the query tensor (device pointer).
 * @param k Pointer to the key tensor (device pointer).
 * @param rope_freqs Pointer to the RoPE frequencies (device pointer).
 * @param B Batch size.
 * @param NH Number of attention heads.
 * @param T Sequence length.
 * @param HS Head size (dimension of each attention head).
 * @param is_forward Flag indicating forward or backward.
 * @param scaling RoPE scaling factor to adjust for longer sequences.
 */
__global__ void apply_rope_kernel(floatX* q, floatX* k, float* rope_freqs, 
                                    int B, int NH, int T, int HS, 
                                    bool is_forward, float scaling) {
    // Calculate the number of elements processed by each thread
    int elements_per_thread = HS / x128::size;

    // Calculate global thread idx
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * NH * T * elements_per_thread) { return; }

    // Calculate indices of batch, ...
    int batch_idx = idx / (NH * T * elements_per_thread);
    int rest = idx % (NH * T * elements_per_thread);
    int head_idx = rest / (T * elements_per_thread);
    rest = rest % (T * elements_per_thread);
    int seq_idx = rest / elements_per_thread;
    int element_idx = rest % elements_per_thread;

    // Calculate the pointer to the relevant RoPE frequencies
    float* freq_ptr = rope_freqs + seq_idx * (HS / 2) + element_idx * (x128::size / 2);
    f128 freq_cache = load128(freq_ptr);

    // Calculate the index for the tensor
    int tensor_idx = batch_idx * NH * T * HS +
                     head_idx * T * HS +
                     seq_idx * HS +
                     element_idx * x128::size;

    // Load query and key values
    x128 query_values = load128cs(&q[tensor_idx]);
    x128 key_values = load128cs(&k[tensor_idx]);
    x128 rotated_query, rotated_key;

    // Apply RoPE with scaling
    for (int k = 0; k < x128::size / 2; k++) { 
        float q1 = query_values[2*k];
        float q2 = query_values[2*k + 1];
        float k1 = key_values[2*k];
        float k2 = key_values[2*k + 1];

        // Calculate scaled frequency
        float scaled_freq = freq_cache[k] / scaling;
        float cos_theta = cosf(scaled_freq);
        float sin_theta = sinf(scaled_freq);

        float direction = is_forward ? (float) 1: (float) -1;

        // See formula 34 in https://arxiv.org/pdf/2104.09864
        rotated_query[2*k] = q1 * cos_theta - direction * q2 * sin_theta;
        rotated_query[2*k + 1] = q2 * cos_theta - direction * q1 * sin_theta;

        rotated_key[2*k] = k1 * cos_theta - direction * k2 * sin_theta;
        rotated_key[2*k + 1] = k2 * cos_theta - direction * k1 * sin_theta;
    }

    store128cs(&q[tensor_idx], rotated_query);
    store128cs(&k[tensor_idx], rotated_key);
}

/**
 * @brief Kernel Function: Initializes rotary positional embedding frequencies.
 *
 * @param rope_freqs RoPE frequencies
 * @param rope_base_freq The base frequency for RoPE
 * 
 */
__global__ void init_rope_freqs_kernel(float* rope_freqs, float rope_base_freq) {
    int freqs_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Compute the rotary positional encoding frequency based on the method 
    // described in Section 3.2.2 of "RoFormer: Enhanced Transformer with Rotary Position Embedding" 
    // (https://arxiv.org/pdf/2104.09864).
    int m = blockIdx.x;

    float d = 2.f * blockDim.x;
    int i = threadIdx.x + 1;
    float theta_i = __powf(rope_base_freq, -2.0f * (i - 1) / d);

    rope_freqs[freqs_idx] = (float)m * theta_i;
}

// ----------------------------------------------------------------------------
// kernel launchers

/**
 * @brief Initializes the RoPE (Rotary Position Embedding) context.
 *
 * @param context context of RoPE
 * @param base The base frequency for RoPE
 * @param scaling The scaling factor for RoPE
 * @param cuda_stream CUDA stream in which the kernel will be executed.
 */
void init_rope_context(RoPEContext** context, float base, float scaling, cudaStream_t cuda_stream) {   
    *context = (RoPEContext*) malloc(sizeof(RoPEContext));
    (*context)->base_frequency = base;
    (*context)->scaling = scaling;
}

/**
 * @brief Host Function: Launches the kernel to initialize RoPE frequencies.
 *
 * @param context context of RoPE
 * @param HS The size of each attention head (number of dimensions).
 * @param cuda_stream CUDA stream in which the kernel will be executed.
 */
void init_rope_freqs(RoPEContext** context, int max_seq_len, int HS, cudaStream_t stream) {
    cudaCheck(cudaMalloc(&((*context)->rope_freqs), max_seq_len * (HS / 2) * sizeof(float))); 
    NVTX_RANGE_FN();
    init_rope_freqs_kernel<<<max_seq_len, HS / 2, 0, stream>>>((*context)->rope_freqs, (*context)->base_frequency); 
    cudaCheck(cudaGetLastError());
}

#endif // ROPE_CUH