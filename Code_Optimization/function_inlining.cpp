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

//function inlining�� ���� ���� �ڵ�� function inlining�� �� �ڵ��� ����ð��� ��
int main(void)
{
	float compute_time1 = 0;//function inlining�� ���� ���� �ڵ��� ����ð� ����
	float compute_time2 = 0;//function inlining�� �� �ڵ��� ����ð� ����

	//sum�� �� �迭���� �ʱ�ȭ
	float* a = (float*)malloc(sizeof(float) * N * N);
	float* b = (float*)malloc(sizeof(float) * N * N);
	for (int i = 0; i < N * N; i++)
	{
		a[i] = 2.4;
		b[i] = 1.1;
	}

	float result1 = 0.0;//function inlining�� ���� ���� �ڵ��� ����� ����
	float result2 = 0.0;//function inlining�� �� �ڵ��� ����� ����

	//function inlining�� ���� ���� �ڵ��� ���
	CHECK_TIME_START;//�ð� üũ ����
	without_function_inlining(a, b, &result1);
	CHECK_TIME_END(compute_time1);//�ð�üũ ����

	//function inlining�� �� �ڵ��ǰ��
	CHECK_TIME_START;//�ð� üũ ����
	with_function_inlining(a, b, &result2);
	CHECK_TIME_END(compute_time2);//�ð�üũ ����

	check_results(result1, result2);

	printf("\n\n����ð�\n");
	printf("function_inlining�� ���� ���� �ڵ�: %f ms\n", compute_time1);
	printf("function_inlining�� �� �ڵ�: %f ms\n", compute_time2);

	free(a);
	free(b);

}


//�� result�� ���� ���� ������, �� ���� ����� ������ ������ check
void check_results(float result1, float result2)
{
	if (result1 == result2)
		printf("Results are same\n");
	else
		printf("Results are not same\n");
}

//a�� b���� ���ؼ� ����� ��ȯ�ϴ� �Լ�
float add_func(float a, float b)
{
	float ret = a + b;
	return ret;
}

//function_inlining�� ���� ���� �ڵ�
void without_function_inlining(float* a, float* b, float* result)
{
	for (int i = 0; i < pow(N, 2); i++)
	{
		*result += add_func(a[i], b[i]);
	}
}
//function_inlining�� �� �ڵ��� ���
void with_function_inlining(float* a, float* b, float* result)
{
	for (int i = 0; i < pow(N, 2); i++)
	{
		*result += a[i] + b[i];
	}
}

