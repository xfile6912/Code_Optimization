#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include<stdio.h>
#include<stdlib.h>
#include <math.h>
#include <Windows.h>
#include <time.h>
#include <assert.h>


#define CUDA_CALL(x) { const cudaError_t a = (x); if(a != cudaSuccess) { printf("\nCuda Error: %s (err_num=%d) at line:%d\n", cudaGetErrorString(a), a, __LINE__); cudaDeviceReset(); assert(0);}}
typedef float TIMER_T;

//CPU 시간 측정
__int64 start, freq, end;
#define CHECK_TIME_START { QueryPerformanceFrequency((LARGE_INTEGER*)&freq); QueryPerformanceCounter((LARGE_INTEGER*)&start); }
#define CHECK_TIME_END(a) { QueryPerformanceCounter((LARGE_INTEGER*)&end); a = (float)((float)(end - start) / (freq / 1000.0f)); }

//GPU 시간 측정 함수
cudaEvent_t cuda_timer_start, cuda_timer_stop;
#define CUDA_STREAM_0 (0)

void create_gpu_timer()
{
    CUDA_CALL(cudaEventCreate(&cuda_timer_start));
    CUDA_CALL(cudaEventCreate(&cuda_timer_stop));
}

void destroy_gpu_timer()
{
    CUDA_CALL(cudaEventDestroy(cuda_timer_start));
    CUDA_CALL(cudaEventDestroy(cuda_timer_stop));
}

inline void start_gpu_timer()
{
    cudaEventRecord(cuda_timer_start, CUDA_STREAM_0);
}

inline TIMER_T stop_gpu_timer()
{
    TIMER_T ms;
    cudaEventRecord(cuda_timer_stop, CUDA_STREAM_0);
    cudaEventSynchronize(cuda_timer_stop);

    cudaEventElapsedTime(&ms, cuda_timer_start, cuda_timer_stop);
    return ms;
}
//gpu 시간 측정 함수
#define CHECK_TIME_INIT_GPU() { create_gpu_timer(); }
#define CHECK_TIME_START_GPU() { start_gpu_timer(); }
#define CHECK_TIME_END_GPU(a) { a = stop_gpu_timer(); }
#define CHECK_TIME_DEST_GPU() { destroy_gpu_timer(); }

TIMER_T cpu_time = 0;
TIMER_T gpu_time = 0;



typedef struct {
    int width;
    int height;
    float* elements;
} Array;


#define MAX_N_ELEMENTS	(1 << 28)

void init_array(float* array, int n) {

    int i;
    //array의 각 원소에 float값을 생성해 줌
    for (i = 0; i < n; i++) {
        array[i] = i + 3.14f;
    }
}
void sum_arrays(float* a, float* b, float* c, int n) {
    int i;

    //array a의 합과 array b의 합을 c에 저장
    for (i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

__global__ void SumArraysKernel(Array A, Array B, Array C) {
    //계산이 수행어야 하는 task의 id(Array 내에서의 index라고 보면 됨)를 구해줌
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int id = gridDim.x * blockDim.x * row + col;
    //Array A와 Array B의 합을 c에 저장
    C.elements[id] = A.elements[id] + B.elements[id];
}

cudaError_t sum_arrays_GPU(const Array A, const Array B, Array C);

int BLOCK_SIZE =64;//BLOCK당 THREAD의 개수 설정

int main()
{
    int n_elements;

    srand((unsigned int)time(NULL));
    n_elements = MAX_N_ELEMENTS;
    Array A, B, C_RESULT, G_RESULT;
    A.width = B.width = C_RESULT.width = G_RESULT.width = (1<<13);
    A.height = B.height = C_RESULT.height = G_RESULT.height = MAX_N_ELEMENTS / (1<<13);

    //Array를 위한 공간 할당
    A.elements = (float*)malloc(sizeof(float) * MAX_N_ELEMENTS);
    B.elements = (float*)malloc(sizeof(float) * MAX_N_ELEMENTS);
    C_RESULT.elements = (float*)malloc(sizeof(float) * MAX_N_ELEMENTS);
    G_RESULT.elements = (float*)malloc(sizeof(float) * MAX_N_ELEMENTS);

    //Array의 원소들 초기화
    init_array(A.elements, MAX_N_ELEMENTS);
    init_array(B.elements, MAX_N_ELEMENTS);

    //CPU Array Sum 작업 시작 및 시간 측정
    CHECK_TIME_START;
    sum_arrays(A.elements, B.elements, C_RESULT.elements, n_elements);
    CHECK_TIME_END(cpu_time);

    //GPU Array Sum 작업 수행
    cudaError_t cudaStatus = sum_arrays_GPU(A, B, G_RESULT);
    //GPU Array Sum 작업 실패시
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "GPU 계산 실패\n");
        return 1;
    }

    //CUDA DEVICE Reset 실행
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Cuda Device Reset 실패\n");
        return 1;
    }

    printf("GPU_RESULT[10] = %f\n", G_RESULT.elements[10]);
    printf("걸린시간 : %.6f ms\n\n", gpu_time);

    return 0;
}
cudaError_t sum_arrays_GPU(const Array A, const Array B, Array C) {

    cudaError_t cudaStatus;
    //GPU 선택(멀티 GPU사용시 값 변경해야함)
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "GPU 선택에 실패");
        goto Error;
    }

    Array gpu_A, gpu_B, gpu_C;
    size_t size;

    //A Array의 내용을 복사
    gpu_A.width = A.width; gpu_A.height = A.height;
    size = A.width * A.height * sizeof(float);
    //CUDA MEMORY 할당
    CUDA_CALL(cudaMalloc(&gpu_A.elements, size))
        CUDA_CALL(cudaMemcpy(gpu_A.elements, A.elements, size, cudaMemcpyHostToDevice))

        //B Array의 내용을 복사
        gpu_B.width = B.width; gpu_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    //CUDA MEMORY 할당
    CUDA_CALL(cudaMalloc(&gpu_B.elements, size))
        CUDA_CALL(cudaMemcpy(gpu_B.elements, B.elements, size, cudaMemcpyHostToDevice))

        //결과를 저장할 CUDA MEMORY 할당
        gpu_C.width = C.width; gpu_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    CUDA_CALL(cudaMalloc(&gpu_C.elements, size))

        //BLOCK과 GLID의 dimension 설정
        dim3 dimBlock(8, BLOCK_SIZE/8);
    dim3 dimGrid(A.width / dimBlock.x, A.height / dimBlock.y);

    //시간 측정 시작
    CHECK_TIME_INIT_GPU()
        CHECK_TIME_START_GPU()

        //GPU 계산작업
        SumArraysKernel << < dimGrid, dimBlock >> > (gpu_A, gpu_B, gpu_C);

    //시간측정 종료
    CHECK_TIME_END_GPU(gpu_time)
        CHECK_TIME_DEST_GPU()

        CUDA_CALL(cudaGetLastError())
        CUDA_CALL(cudaDeviceSynchronize())

        //계산한 결과를 CudaMemory로부터 Host의 Memory로 복사
        CUDA_CALL(cudaMemcpy(C.elements, gpu_C.elements, size, cudaMemcpyDeviceToHost))




    Error:
    cudaFree(gpu_A.elements);
    cudaFree(gpu_B.elements);
    cudaFree(gpu_C.elements);
    return cudaStatus;
}