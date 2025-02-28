#include <iostream>
#include <cuda.h>
//you can change the grid_size
#define GRID_SIZE 32
//you can change the block_size
#define BLOCK_SIZE 32

#define MIN(a, b) ((a)<(b) ? (a):(b))

#define TILE_SIZE 2
__global__ void cnn00(int N,int C,int K,int H,int W,int R, int S, int u, int v, int P, int Q,
               float *d_input, float * d_weight, float * d_output){
    //@@ cnn kernel design
    int n = blockIdx.x;
    int k = blockIdx.y;
    int q0 = threadIdx.x;
    int p0 = threadIdx.y;
    
    for (int p = p0; p < P; p += TILE_SIZE) {
        int ij = p * u;
        for (int q = q0; q < Q; q += TILE_SIZE) {
            int ii = q * v;
            float sum = 0.0;
            for (int c = 0; c < C; c++) {
                for (int r = 0; r < R; r++) {
                    for (int s = 0; s < S; s++) {
                        sum += d_input[n*C*H*W + c*H*W + (ij+r)*W + ii + s] * d_weight[k*C*R*S + c*R*S + r*S + s];
                    }
                }
            }
            d_output[n*K*P*Q + k*P*Q + p*Q + q] = sum;
        }
    }
}

__global__ void cnn01(int N,int C,int K,int H,int W,int R, int S, int u, int v, int P, int Q,
    float *d_input, float * d_weight, float * d_output){
    //@@ cnn kernel design
    int n = blockIdx.x;
    int k = blockIdx.y;
    int q0 = threadIdx.x;
    int p0 = threadIdx.y;

    __shared__ float weight_buf[TILE_SIZE][TILE_SIZE];

    for (int p = p0; p < P; p += TILE_SIZE) {
        int ij = p * u;
        for (int q = q0; q < Q; q += TILE_SIZE) {
            int ii = q * v;
            float sum = 0.0;
            for (int c = 0; c < C; c++) {
                int r_round_bound = (R + TILE_SIZE - 1) / TILE_SIZE;
                int s_round_bound = (S + TILE_SIZE - 1) / TILE_SIZE;
                for (int r_round = 0; r_round < r_round_bound; r_round++) {
                    for (int s_round = 0; s_round < s_round_bound; s_round++) {
                        int r_index = r_round*TILE_SIZE+p0;
                        int s_index = s_round*TILE_SIZE+q0;
                        if ((r_index < R) && (s_index < S)) {
                            weight_buf[p0][q0] = d_weight[k*C*R*S + c*R*S + r_index*S + s_index];
                        }
                        __syncthreads();
                        
                        int r_begin = r_round * TILE_SIZE;
                        int r_end = MIN(r_begin + TILE_SIZE, R);
                        int s_begin = s_round * TILE_SIZE;
                        int s_end = MIN(s_begin + TILE_SIZE, S);
                        for (int i_r = r_begin; i_r < r_end; ++i_r) {
                            for (int i_s = s_begin; i_s < s_end; ++i_s) {
                                sum += d_input[n*C*H*W + c*H*W + (ij+i_r)*W + ii + i_s]*weight_buf[i_r-r_begin][i_s-s_begin];
                            }
                        }
                        __syncthreads();
                    }
                }
            }
            d_output[n*K*P*Q + k*P*Q + p*Q + q] = sum;
            /*
            for (int c = 0; c < C; c++) {
                for (int r = 0; r < R; r++) {
                    for (int s = 0; s < S; s++) {
                        // load shared memory


                        //sum += d_input[n*C*H*W + c*H*W + (ij+r)*W + ii + s] * d_weight[k*C*R*S + c*R*S + r*S + s];
                    }
                }
            }
            */
        }
    }
}





__global__ void cnn001(int N,int C,int K,int H,int W,int R, int S, int u, int v, int P, int Q,
    float *d_input, float * d_weight, float * d_output){
    int n = blockIdx.x;
    int k = blockIdx.y;
    int q0 = threadIdx.x;
    int p0 = threadIdx.y;
    int c0 = threadIdx.z;
    for (int p = p0; p < P; p += TILE_SIZE) {
        int ij = p * u;
        for (int q = q0; q < Q; q += TILE_SIZE) {
            int ii = q * v;
            float _sum = 0.0;
            for (int c = c0; c < C; c += TILE_SIZE) {
                for (int r = 0; r < R; r++) {
                    for (int s = 0; s < S; s++) {
                        _sum += d_input[n*C*H*W + c*H*W + (ij+r)*W + ii + s] * d_weight[k*C*R*S + c*R*S + r*S + s];
                    }
                }
            }
            atomicAdd(&d_output[n*K*P*Q + k*P*Q + p*Q + q], _sum);
        }
    }
}

__global__ void cnn002(int N,int C,int K,int H,int W,int R, int S, int u, int v, int P, int Q,
    float *d_input, float * d_weight, float * d_output){
    int n = blockIdx.x;
    int k = blockIdx.y;
    int q0 = threadIdx.x;
    int p0 = threadIdx.y;
    int c0 = threadIdx.z;

    __shared__ float wbuf[TILE_SIZE][TILE_SIZE][TILE_SIZE];

    for (int p = p0; p < P; p += TILE_SIZE) {
        int ij = p * u;
        for (int q = q0; q < Q; q += TILE_SIZE) {
            int ii = q * v;
            float _sum = 0.0;
            int c_lim = (C + TILE_SIZE - 1) / TILE_SIZE;
            int r_lim = (R + TILE_SIZE - 1) / TILE_SIZE;
            int s_lim = (S + TILE_SIZE - 1) / TILE_SIZE;

            for (int c_round = 0; c_round < c_lim; ++c_round) {
                for (int r_round = 0; r_round < r_lim; ++r_round) {
                    for (int s_round = 0; s_round < s_lim; ++s_round) {
                        int c_begin = c_round*TILE_SIZE;
                        int c_end = MIN(c_begin+TILE_SIZE, C);
                        int r_begin = r_round*TILE_SIZE;
                        int r_end = MIN(r_begin+TILE_SIZE, R);
                        int s_begin = s_round*TILE_SIZE;
                        int s_end = MIN(s_begin+TILE_SIZE, S);

                        int c_index = c_begin + c0;
                        int r_index = r_begin + p0;
                        int s_index = s_begin + q0;
                        if ((c_index < C) && (r_index < R) && (s_index < S)) {
                            wbuf[c0][p0][q0] = d_weight[k*C*R*S + c_index*R*S + r_index*S + s_index];
                        }
                        __syncthreads();

                        for (int c = c_begin+c0; c < c_end; c+=TILE_SIZE) {
                            for (int r = r_begin; r < r_end; ++r) {
                                for (int s = s_begin; s < s_end; ++s) {
                                    _sum += d_input[n*C*H*W + c*H*W + (ij+r)*W + ii + s] * wbuf[c-c_begin][r-r_begin][s-s_begin];
                                }
                            }
                        }
                        __syncthreads();
                    }
                }
            }
            atomicAdd(&d_output[n*K*P*Q + k*P*Q + p*Q + q], _sum);
        }
    }
}


__global__ void cnn003(int N,int C,int K,int H,int W,int R, int S, int u, int v, int P, int Q,
    float *d_input, float * d_weight, float * d_output){
    int n = blockIdx.x;
    int k = blockIdx.y;
    int q0 = threadIdx.x;
    int p0 = threadIdx.y;
    int c0 = threadIdx.z;

    __shared__ float wbuf[TILE_SIZE][TILE_SIZE][TILE_SIZE];
    __shared__ float part_sum[TILE_SIZE][TILE_SIZE][TILE_SIZE];

    for (int p = p0; p < P; p += TILE_SIZE) {
        int ij = p * u;
        for (int q = q0; q < Q; q += TILE_SIZE) {
            int ii = q * v;
            float _sum = 0.0;
            int c_lim = (C + TILE_SIZE - 1) / TILE_SIZE;
            int r_lim = (R + TILE_SIZE - 1) / TILE_SIZE;
            int s_lim = (S + TILE_SIZE - 1) / TILE_SIZE;

            part_sum[c0][p0][q0] = 0.0;

            for (int c_round = 0; c_round < c_lim; ++c_round) {
                for (int r_round = 0; r_round < r_lim; ++r_round) {
                    for (int s_round = 0; s_round < s_lim; ++s_round) {
                        int c_begin = c_round*TILE_SIZE;
                        int c_end = MIN(c_begin+TILE_SIZE, C);
                        int r_begin = r_round*TILE_SIZE;
                        int r_end = MIN(r_begin+TILE_SIZE, R);
                        int s_begin = s_round*TILE_SIZE;
                        int s_end = MIN(s_begin+TILE_SIZE, S);

                        int c_index = c_begin + c0;
                        int r_index = r_begin + p0;
                        int s_index = s_begin + q0;
                        if ((c_index < C) && (r_index < R) && (s_index < S)) {
                            wbuf[c0][p0][q0] = d_weight[k*C*R*S + c_index*R*S + r_index*S + s_index];
                            //printf("Load d_weight=%f c_index=%d r_index=%d s_index=%d\n", d_weight[k*C*R*S + c_index*R*S + r_index*S + s_index],\
                            c_index, r_index, s_index);
                        }
                        __syncthreads();

                        for (int c = c_begin+c0; c < c_end; c+=TILE_SIZE) {
                            for (int r = r_begin; r < r_end; ++r) {
                                for (int s = s_begin; s < s_end; ++s) {
                                    _sum += d_input[n*C*H*W + c*H*W + (ij+r)*W + ii + s] * wbuf[c%TILE_SIZE][r-r_begin][s-s_begin];
                                }
                            }
                        }
                        __syncthreads();
                    }
                }
            }
            //atomicAdd(&d_output[n*K*P*Q + k*P*Q + p*Q + q], _sum);
            part_sum[c0][p0][q0] = _sum;
            __syncthreads();
            if (c0 == 0) {
                for (int iter = 1; iter < TILE_SIZE; ++iter) {
                    _sum += part_sum[iter][p0][q0];
                }
                d_output[n*K*P*Q + k*P*Q + p*Q + q] = _sum;
            }
        }
    }
}


