#include <iostream>
#include <cuda.h>
//you can change the grid_size
#define GRID_SIZE 32
//you can change the block_size
#define BLOCK_SIZE 32

#define TILE_SIZE 8
__global__ void cnn0(int N,int C,int K,int H,int W,int R, int S, int u, int v, int P, int Q,
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

__global__ void cnn1(int N,int C,int K,int H,int W,int R, int S, int u, int v, int P, int Q,
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

__global__ void cnn2(int N,int C,int K,int H,int W,int R, int S, int u, int v, int P, int Q,
    float *d_input, float * d_weight, float * d_output){
    int n = blockIdx.x;
    int k = blockIdx.y;
    int q0 = threadIdx.x;
    int p0 = threadIdx.y;
    int c0 = threadIdx.z;

    __shared__ float weight_buf[TILE_SIZE][TILE_SIZE][TILE_SIZE];
    for (int p = p0; p < P; p += TILE_SIZE) {
        int ij = p * u;
        for (int q = q0; q < Q; q += TILE_SIZE) {
            int ii = q * v;
            
            for (int c = c0; c < C; c += TILE_SIZE) {
                for (int r = p0; r < R; r += TILE_SIZE) {
                    for (int s = q0; s < S; s += TILE_SIZE) {
                        weight_buf[c%TILE_SIZE][r%TILE_SIZE][s%TILE_SIZE] = d_weight[k*C*R*S + c*R*S + r*S + s];
                        __syncthreads();

                        float sum = 0.0;
                        int r_begin = (r / TILE_SIZE) * TILE_SIZE;
                        int r_end = r_begin + TILE_SIZE;
                        int s_begin = (s / TILE_SIZE) * TILE_SIZE;
                        int s_end = s_begin + TILE_SIZE;
                        for (int i_r = r_begin; i_r < r_end; ++i_r) {
                            for (int i_s = s_begin; i_s < s_end; ++i_s) {
                                sum += d_input[n*C*H*W + c*H*W + (ij+i_r)*W + ii + i_s] * weight_buf[c%TILE_SIZE][i_r - r_begin][i_s - s_begin];
                            }
                        }
                        atomicAdd(&d_output[n*K*P*Q + k*P*Q + p*Q + q], sum);
                        __syncthreads();
                    }
                }
            }
        }
    }
}


/*
for(unsigned int c=0; c<C; c ++) { // input feature map
for(unsigned int p=0; p<P; p ++) { // output height
 unsigned int ij = p * u; // input height
 for (unsigned int q = 0; q<Q; q ++) { // output width
     unsigned int ii = q * v; // input width
     for (unsigned int r = 0; r<R; r ++) { // filter height
         for (unsigned int s = 0; s < S; s ++) {// filter width
             //output_seq[n][k][p][q] += input [n][c][ij+r][ii+s] * weight[k][c][r][s];
             output_seq[n*K*P*Q + k*P*Q + p*Q + q] += input[n*C*H*W + c*H*W + (ij+r)*W + ii+s] * weight[k*C*R*S+c*R*S+r*S+s];
         }
     }
 }
}
}
*/

