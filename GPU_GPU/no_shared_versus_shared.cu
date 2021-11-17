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
#define USE_GPU_TIMER 1

#define	IN
#define OUT
#define INOUT


#if USE_GPU_TIMER == 1
cudaEvent_t cuda_timer_start, cuda_timer_stop;
#define CUDA_STREAM_0 (0)


void create_device_timer()
{
	CUDA_CALL(cudaEventCreate(&cuda_timer_start));
	CUDA_CALL(cudaEventCreate(&cuda_timer_stop));
}

void destroy_device_timer()
{
	CUDA_CALL(cudaEventDestroy(cuda_timer_start));
	CUDA_CALL(cudaEventDestroy(cuda_timer_stop));
}

inline void start_device_timer()
{
	cudaEventRecord(cuda_timer_start, CUDA_STREAM_0);
}

inline TIMER_T stop_device_timer()
{
	TIMER_T ms;
	cudaEventRecord(cuda_timer_stop, CUDA_STREAM_0);
	cudaEventSynchronize(cuda_timer_stop);

	cudaEventElapsedTime(&ms, cuda_timer_start, cuda_timer_stop);
	return ms;
}

#define CHECK_TIME_INIT_GPU() { create_device_timer(); }
#define CHECK_TIME_START_GPU() { start_device_timer(); }
#define CHECK_TIME_END_GPU(a) { a = stop_device_timer(); }
#define CHECK_TIME_DEST_GPU() { destroy_device_timer(); }
#else
#define CHECK_TIME_INIT_GPU()
#define CHECK_TIME_START_GPU()
#define CHECK_TIME_END_GPU(a)
#define CHECK_TIME_DEST_GPU()
#endif

#define N_SIZE (1 << 26)													// 전체 데이터 사이즈
#define NF_SIZE 2048													    // Nf 크기

#define NO_SHARED 0															// shared memory를 사용하지 않는 커널 실행 flag
#define SHARED 1															// shared memory를 사용하는 커널 실행 flag

#define BLOCK_SIZE 256													// CUDA 커널 thread block 사이즈

#define BLOCK_WIDTH 16
#define BLOCK_HEIGHT (BLOCK_SIZE / BLOCK_WIDTH)

extern __shared__ int shared_buffer[];								//Shared Memory 동적 할당
TIMER_T device_time = 0;

int N;
int Nf;

int* arr;
int* sum_no_shared;
int* sum_shared;

cudaError_t sum_array_gpu(IN int* p_arr, OUT int* p_sum_gpu, int Nf, int shared_flag);


//배열의 index-Nf 부터 index+Nf 데이터 까지의 합을 계산하는 GPU코드(Shared Memory를 사용하지 않음)
__global__ void sum_array_kernel_no_shared(IN int* d_arr, OUT int* d_sum_gpu, int N, int Nf) {
	const unsigned block_id = blockIdx.y * gridDim.x + blockIdx.x;
	const unsigned thread_id = threadIdx.y * blockDim.x + threadIdx.x;
	const unsigned id = block_id * BLOCK_SIZE + thread_id;

	//sum이라는 local변수를 사용하지 않은 이유는 sum local변수를 이용하게되면 

	for (int i = -Nf; i <= Nf; i++) {
		if (id + i >= N || id + i < 0) continue;
		d_sum_gpu[id] += d_arr[id + i];
	}
}

//배열의 index-Nf 부터 index+Nf 데이터 까지의 합을 계산하는 GPU코드(Shared Memory를 사용)
__global__ void sum_array_kernel_shared(IN int* d_arr, OUT int* d_sum_gpu, int N, int Nf) {
	const unsigned block_id = blockIdx.y * gridDim.x + blockIdx.x;
	const unsigned thread_id = threadIdx.y * blockDim.x + threadIdx.x;
	const unsigned id = block_id * BLOCK_SIZE + thread_id;
	int i;

	//threadid가 0인 경우는 shared memory의 왼쪽 끝 부분을 채워 줘야 함
	if (thread_id == 0)
	{
		for (i = 0; i < Nf; i++)
		{
			if (id + i < Nf) shared_buffer[i] = 0;
			else shared_buffer[i] = d_arr[id + i - Nf];
		}
	}
	//threadid가 BLOCK_SIZE-1인 경우는 shared memory의 오른쪽 끝 부분을 채워줘야 함
	if (thread_id == BLOCK_SIZE - 1)
	{
		for (i = 1; i <= Nf; i++)
		{
			if (id + i >= N) shared_buffer[thread_id + i + Nf] = 0;
			else shared_buffer[thread_id + i + Nf] = d_arr[id + i];
		}
	}
	shared_buffer[thread_id + Nf] = d_arr[id];
	__syncthreads();
	for (i = 0; i <= 2 * Nf; i++)
	{
		d_sum_gpu[id] += shared_buffer[thread_id + i];
	}
}


//데이터 파일을 읽는 코드로 파일에는 순서대로 N, Nf의 크기, N개의 int형 데이터가 저장되어 있음
void read_file() {
	printf("데이터 파일 읽기 시작\n");
	FILE* fp = fopen("data_file.bin", "rb");
	fread(&N, sizeof(int), 1, fp);
	fread(&Nf, sizeof(int), 1, fp);

	arr = (int*)malloc(N * sizeof(int));
	sum_no_shared = (int*)malloc(N * sizeof(int));
	sum_shared = (int*)malloc(N * sizeof(int));

	fread(arr, sizeof(int), N, fp);

	fclose(fp);
	printf("데이터 파일 읽기 완료\n\n");
}

void create_file(IN int n, IN int nf) {
	printf("데이터 파일 생성 시작\n");
	srand((unsigned)time(NULL));
	FILE* fp = fopen("data_file.bin", "wb");
	fwrite(&n, sizeof(int), 1, fp);
	fwrite(&nf, sizeof(int), 1, fp);

	int i, input;

	for (i = 0; i < n; i++) {
		input = (int)((float)rand() / RAND_MAX * 200 - 100);
		fwrite(&input, sizeof(int), 1, fp);
	}

	fclose(fp);
	printf("데이터 파일 생성 완료\n\n");
}

int main()
{
	int i;
	create_file(N_SIZE, NF_SIZE);
	read_file();

	TIMER_T time_no_shared = 0.0f, time_shared = 0.0f;

	//Shared Memory를 이용하지 않는 경우
	sum_array_gpu(arr, sum_no_shared, Nf, NO_SHARED);
	time_no_shared = device_time;

	//Shared Memory를 이용하는 경우
	sum_array_gpu(arr, sum_shared, Nf, SHARED);
	time_shared = device_time;

	for (i = 0; i < N; i++) {
		//두 결과가 다른경우
		if (sum_no_shared[i] != sum_shared[i]) {
			printf("ERROR: 서로 다른 결과가 나옴\n");
			return 0;
		}
	}

	printf("Shared Memory를 이용하지 않은 경우 : %.6f ms\n", time_no_shared);
	printf("Shared Memory를 이용한 경우 : % .6f ms\n", time_shared);

	free(arr);
	free(sum_no_shared);
	free(sum_shared);

	return 0;
}

//FLAG값에 맞는 커널을 실행
cudaError_t sum_array_gpu(IN int* p_arr, OUT int* p_sum_gpu, int Nf, int shared_flag) {
	cudaError_t cudaStatus;

	//GPU 선택(멀티 GPU사용시 값 변경해야함)
	CUDA_CALL(cudaSetDevice(0));

	int* d_arr, * d_sum;
	size_t mem_size;

	mem_size = N * sizeof(int);
	//cuda 메모리 할당
	CUDA_CALL(cudaMalloc(&d_arr, mem_size));
	CUDA_CALL(cudaMalloc(&d_sum, mem_size));


	CUDA_CALL(cudaMemcpy(d_arr, p_arr, mem_size, cudaMemcpyHostToDevice));

	//BLOCK과 GLID의 dimension 설정
	dim3 blockDIm(BLOCK_WIDTH, BLOCK_HEIGHT);
	dim3 gridDim(N / BLOCK_SIZE);

	//시간 측정 시작
	CHECK_TIME_INIT_GPU();
	CHECK_TIME_START_GPU();
	switch (shared_flag)
	{
		//flag에 따라 no_shared 실행 시켜줌
	case NO_SHARED:
		sum_array_kernel_no_shared << <gridDim, blockDIm >> > (d_arr, d_sum, N, Nf);
		break;
		//flag에 따라 shared 실행시켜줌
		//sizeof(int)*(BLOCK_SIZE*2*NF)는 Shared Memory의 크기를 동적으로 할당해주는 것
	case SHARED:
		sum_array_kernel_shared << <gridDim, blockDIm, sizeof(int)* (BLOCK_SIZE + 2 * Nf) >> > (d_arr, d_sum, N, Nf);
		break;

	}

	CUDA_CALL(cudaStatus = cudaDeviceSynchronize());
	CHECK_TIME_END_GPU(device_time);
	CHECK_TIME_DEST_GPU();

	CUDA_CALL(cudaMemcpy(p_sum_gpu, d_sum, mem_size, cudaMemcpyDeviceToHost));

	cudaFree(d_arr);
	cudaFree(d_sum);

	return cudaStatus;
}
