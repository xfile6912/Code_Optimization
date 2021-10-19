#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <Windows.h>

__int64 start, freq, end;
#define CHECK_TIME_START QueryPerformanceFrequency((LARGE_INTEGER*)&freq); QueryPerformanceCounter((LARGE_INTEGER*)&start)
#define CHECK_TIME_END(a) QueryPerformanceCounter((LARGE_INTEGER*)&end); a = (float)((float)(end - start) / (freq / 1000.0f))
#define N 640


void check_results(float result1, float result2);
void without_loop_unrolling(float *a, float* result);
void with_loop_unrolling(float *a, float* result);

//loop unrolling�� ���� ���� �ڵ�� loop unrolling�� �� �ڵ��� ����ð��� ��
int main(void)
{
	float compute_time1 = 0;//loop unrolling�� ���� ���� �ڵ��� ����ð� ����
	float compute_time2 = 0;//loop unrolling�� �� �ڵ��� ����ð� ����

	//sum�� �� �迭�� �ʱ�ȭ
	float *a = (float*)malloc(sizeof(float)*N*N);
	for (int i = 0; i < N * N; i++)
		a[i] = 2.4;
	float result1 = 0.0;//loop unrolling�� ���� ���� �ڵ��� ����� ����
	float result2 = 0.0;//loop unrolling�� �� �ڵ��� ����� ����

	//loop unrolling�� ���� ���� �ڵ��� ���
	CHECK_TIME_START;//�ð� üũ ����
	without_loop_unrolling(a, &result1);
	CHECK_TIME_END(compute_time1);//�ð�üũ ����

	//loop unrolling�� �� �ڵ��ǰ��
	CHECK_TIME_START;//�ð� üũ ����
	with_loop_unrolling(a, &result2);
	CHECK_TIME_END(compute_time2);//�ð�üũ ����

	check_results(result1, result2);

	printf("\n\n����ð�\n");
	printf("loop unrolling�� ���� ���� �ڵ�: %f ms\n", compute_time1);
	printf("loop unrolling�� �� �ڵ�: %f ms\n", compute_time2);


}


//�� result�� ���� ���� ������, �� ���� ����� ������ ������ check
void check_results(float result1, float result2)
{
	if (result1 == result2)
		printf("Results are same\n");
	else
		printf("Results are not same\n");
}

//loop unrolling�� ���� ���� �ڵ�
void without_loop_unrolling(float* a, float* result)
{
	for (int i = 0; i < pow(N, 2); i++)
	{
		*result += a[i];
	}
}
//loop unrolling�� �� �ڵ��� ���
void with_loop_unrolling(float* a, float* result)
{
	for (int i = 0; i < pow(N, 2); i+=4)
	{
		*result += a[i];
		*result += a[i + 1];
		*result += a[i + 2];
		*result += a[i + 3];
	}
}
