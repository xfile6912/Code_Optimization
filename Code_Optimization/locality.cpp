#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <Windows.h>

__int64 start, freq, end;
#define CHECK_TIME_START QueryPerformanceFrequency((LARGE_INTEGER*)&freq); QueryPerformanceCounter((LARGE_INTEGER*)&start)
#define CHECK_TIME_END(a) QueryPerformanceCounter((LARGE_INTEGER*)&end); a = (float)((float)(end - start) / (freq / 1000.0f))
#define N 500

void init_array(float* a, float* b, float* transpose_b, float* result1, float* result2);
void check_results(float* result1, float* result2);
void without_locality(float* a, float* b, float* result);
void with_locality(float* a, float* transpose_b, float* result);



//행렬의 곱셈(a * b)을 통해 locality를 고려한 코드와, locality를 고려하지 않은 코드의 수행시간 비교
int main(void)
{
	float compute_time1 = 0;//locality를 적용하지 않은 코드의 수행시간 저장
	float compute_time2 = 0;//locality를 적용한 코드의 수행시간 저장

	float* a = (float*)malloc(sizeof(float) * N * N);//행렬 a
	float* b = (float*)malloc(sizeof(float) * N * N);//행렬 b
	float* transpose_b = (float*)malloc(sizeof(float) * N * N);//locality를 위해 b를 transpose해준 값

	float* result1 = (float*)malloc(sizeof(float) * N * N);//locality를 적용하지 않은 코드의 결과를 저장
	float* result2 = (float*)malloc(sizeof(float) * N * N);//locality를 적용한 코드의 결과를 저장


	init_array(a, b, transpose_b, result1, result2);//행렬들 초기화

	//locality를 적용하지 않은 경우
	CHECK_TIME_START;//시간 체크 시작
	without_locality(a, b, result1);
	CHECK_TIME_END(compute_time1);//시간체크 종료

	//locality를 적용한 경우
	CHECK_TIME_START;//시간 체크 시작
	with_locality(a, transpose_b, result2);
	CHECK_TIME_END(compute_time2);//시간체크 종료

	check_results(result1, result2);

	printf("\n\n수행시간\n");
	printf("locality를 고려하지 않은 코드: %f ms\n", compute_time1);
	printf("locality를 고려한 코드: %f ms\n", compute_time2);

	

	//메모리 deallocate.
	free(a);
	free(b);
	free(transpose_b);
	free(result1);
	free(result2);
}

void init_array(float* a, float* b, float* transpose_b, float *result1, float *result2)
{
	int cnt_a = 1.1;
	int cnt_b = 2.2;

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			a[i * N + j] = cnt_a++;//a 행렬을 임의값을 집어넣어 초기화 해줌
			b[i * N + j] = cnt_b++;//b 행렬을 임의값을 집어넣어 초기화 해줌
			result1[i * N + j] = 0;//result1 행렬을 0으로 초기화해줌
			result2[i * N + j] = 0;//result2 행렬을 0으로 초기화해줌
		}
	}

	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			transpose_b[i * N + j] = b[j * N + i];//b 행렬의 transpose값을 transpose_b에 저장해줌
}


//두 result의 값이 서로 같은지, 즉 같은 결과를 도출해 내는지 check
void check_results(float* result1, float* result2)
{
	int check_flag = 1;
	for (int i = 0; i < N * N; i++)
	{
		if (result1[i] != result2[i])
		{
			check_flag = 0;
			break;
		}
	}


	if (check_flag == 1)
		printf("Results are same\n");
	else
		printf("Results are not same\n");
}

//메모리의 locality를 고려하지 않은 코드
void without_locality(float *a, float *b, float *result)
{
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			for (int k = 0; k < N; k++)
			{
				result[i * N + j] += a[i * N + k] * b[k * N + j];
				//매 iteration마다 b를 접근할 때에, 연속적인 memory 영역을 접근하는 것이 아닌, 띄엄띄엄 접근하게 됨 
				//spatial locality를 고려하지 않음.
			}
		}
	}
}
//메모리의 locality를 고려한 코드
void with_locality(float *a, float *transpose_b, float *result)
{
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			for (int k = 0; k < N; k++)
			{
				result[i * N + j] += a[i * N + k] * transpose_b[j * N + k];
				//transpose_b를 사용하여, 매 iteration마다 b를 접근할 때, 연속적인 memory 영역을 접근할 수 있음
				//spatial locality를 고려
			}
		}
	}
}
