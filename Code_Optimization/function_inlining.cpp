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


void check_results(float result1, float result2);
void without_function_inlining(float* a, float *b, float* result);
void with_function_inlining(float* a, float * b, float* result);
float add_func(float a, float b);

//function inlining를 하지 않은 코드와 function inlining를 한 코드의 수행시간을 비교
int main(void)
{
	float compute_time1 = 0;//function inlining를 하지 않은 코드의 수행시간 저장
	float compute_time2 = 0;//function inlining를 한 코드의 수행시간 저장

	//sum을 할 배열들을 초기화
	float* a = (float*)malloc(sizeof(float) * N * N);
	float* b = (float*)malloc(sizeof(float) * N * N);
	for (int i = 0; i < N * N; i++)
	{
		a[i] = 2.4;
		b[i] = 1.1;
	}

	float result1 = 0.0;//function inlining를 하지 않은 코드의 결과값 저장
	float result2 = 0.0;//function inlining를 한 코드의 결과값 저장

	//function inlining를 하지 않은 코드의 경우
	CHECK_TIME_START;//시간 체크 시작
	without_function_inlining(a, b, &result1);
	CHECK_TIME_END(compute_time1);//시간체크 종료

	//function inlining를 한 코드의경우
	CHECK_TIME_START;//시간 체크 시작
	with_function_inlining(a, b, &result2);
	CHECK_TIME_END(compute_time2);//시간체크 종료

	check_results(result1, result2);

	printf("\n\n수행시간\n");
	printf("function_inlining을 하지 않은 코드: %f ms\n", compute_time1);
	printf("function_inlining을 한 코드: %f ms\n", compute_time2);

	free(a);
	free(b);

}


//두 result의 값이 서로 같은지, 즉 같은 결과를 도출해 내는지 check
void check_results(float result1, float result2)
{
	if (result1 == result2)
		printf("Results are same\n");
	else
		printf("Results are not same\n");
}

//a와 b값을 더해서 결과를 반환하는 함수
float add_func(float a, float b)
{
	float ret = a + b;
	return ret;
}

//function_inlining를 하지 않은 코드
void without_function_inlining(float* a, float* b, float* result)
{
	for (int i = 0; i < pow(N, 2); i++)
	{
		*result += add_func(a[i], b[i]);
	}
}
//function_inlining를 한 코드의 경우
void with_function_inlining(float* a, float* b, float* result)
{
	for (int i = 0; i < pow(N, 2); i++)
	{
		*result += a[i] + b[i];
	}
}

