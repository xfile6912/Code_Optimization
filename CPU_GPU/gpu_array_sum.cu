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

//CPU �ð� ����
__int64 start, freq, end;
#define CHECK_TIME_START { QueryPerformanceFrequency((LARGE_INTEGER*)&freq); QueryPerformanceCounter((LARGE_INTEGER*)&start); }
#define CHECK_TIME_END(a) { QueryPerformanceCounter((LARGE_INTEGER*)&end); a = (float)((float)(end - start) / (freq / 1000.0f)); }

//GPU �ð� ���� �Լ�
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
//gpu �ð� ���� �Լ�
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
    //array�� �� ���ҿ� float���� ������ ��
    for (i = 0; i < n; i++) {
        array[i] = i + 3.14f;
    }
}
void sum_arrays(float* a, float* b, float* c, int n) {
    int i;

    //array a�� �հ� array b�� ���� c�� ����
    for (i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

__global__ void SumArraysKernel(Array A, Array B, Array C) {
    //����� ������ �ϴ� task�� id(Array �������� index��� ���� ��)�� ������
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int id = gridDim.x * blockDim.x * row + col;
    //Array A�� Array B�� ���� c�� ����
    C.elements[id] = A.elements[id] + B.elements[id];
}

cudaError_t sum_arrays_GPU(const Array A, const Array B, Array C);

int BLOCK_SIZE =64;//BLOCK�� THREAD�� ���� ����

int main()
{
    int n_elements;

    srand((unsigned int)time(NULL));
    n_elements = MAX_N_ELEMENTS;
    Array A, B, C_RESULT, G_RESULT;
    A.width = B.width = C_RESULT.width = G_RESULT.width = (1<<13);
    A.height = B.height = C_RESULT.height = G_RESULT.height = MAX_N_ELEMENTS / (1<<13);

    //Array�� ���� ���� �Ҵ�
    A.elements = (float*)malloc(sizeof(float) * MAX_N_ELEMENTS);
    B.elements = (float*)malloc(sizeof(float) * MAX_N_ELEMENTS);
    C_RESULT.elements = (float*)malloc(sizeof(float) * MAX_N_ELEMENTS);
    G_RESULT.elements = (float*)malloc(sizeof(float) * MAX_N_ELEMENTS);

    //Array�� ���ҵ� �ʱ�ȭ
    init_array(A.elements, MAX_N_ELEMENTS);
    init_array(B.elements, MAX_N_ELEMENTS);

    //CPU Array Sum �۾� ���� �� �ð� ����
    CHECK_TIME_START;
    sum_arrays(A.elements, B.elements, C_RESULT.elements, n_elements);
    CHECK_TIME_END(cpu_time);

    //GPU Array Sum �۾� ����
    cudaError_t cudaStatus = sum_arrays_GPU(A, B, G_RESULT);
    //GPU Array Sum �۾� ���н�
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "GPU ��� ����\n");
        return 1;
    }

    //CUDA DEVICE Reset ����
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Cuda Device Reset ����\n");
        return 1;
    }

    printf("GPU_RESULT[10] = %f\n", G_RESULT.elements[10]);
    printf("�ɸ��ð� : %.6f ms\n\n", gpu_time);

    return 0;
}
cudaError_t sum_arrays_GPU(const Array A, const Array B, Array C) {

    cudaError_t cudaStatus;
    //GPU ����(��Ƽ GPU���� �� �����ؾ���)
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "GPU ���ÿ� ����");
        goto Error;
    }

    Array gpu_A, gpu_B, gpu_C;
    size_t size;

    //A Array�� ������ ����
    gpu_A.width = A.width; gpu_A.height = A.height;
    size = A.width * A.height * sizeof(float);
    //CUDA MEMORY �Ҵ�
    CUDA_CALL(cudaMalloc(&gpu_A.elements, size))
        CUDA_CALL(cudaMemcpy(gpu_A.elements, A.elements, size, cudaMemcpyHostToDevice))

        //B Array�� ������ ����
        gpu_B.width = B.width; gpu_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    //CUDA MEMORY �Ҵ�
    CUDA_CALL(cudaMalloc(&gpu_B.elements, size))
        CUDA_CALL(cudaMemcpy(gpu_B.elements, B.elements, size, cudaMemcpyHostToDevice))

        //����� ������ CUDA MEMORY �Ҵ�
        gpu_C.width = C.width; gpu_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    CUDA_CALL(cudaMalloc(&gpu_C.elements, size))

        //BLOCK�� GLID�� dimension ����
        dim3 dimBlock(8, BLOCK_SIZE/8);
    dim3 dimGrid(A.width / dimBlock.x, A.height / dimBlock.y);

    //�ð� ���� ����
    CHECK_TIME_INIT_GPU()
        CHECK_TIME_START_GPU()

        //GPU ����۾�
        SumArraysKernel << < dimGrid, dimBlock >> > (gpu_A, gpu_B, gpu_C);

    //�ð����� ����
    CHECK_TIME_END_GPU(gpu_time)
        CHECK_TIME_DEST_GPU()

        CUDA_CALL(cudaGetLastError())
        CUDA_CALL(cudaDeviceSynchronize())

        //����� ����� CudaMemory�κ��� Host�� Memory�� ����
        CUDA_CALL(cudaMemcpy(C.elements, gpu_C.elements, size, cudaMemcpyDeviceToHost))




    Error:
    cudaFree(gpu_A.elements);
    cudaFree(gpu_B.elements);
    cudaFree(gpu_C.elements);
    return cudaStatus;
}