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



//����� ����(a * b)�� ���� locality�� ����� �ڵ��, locality�� ������� ���� �ڵ��� ����ð� ��
int main(void)
{
	float compute_time1 = 0;//locality�� �������� ���� �ڵ��� ����ð� ����
	float compute_time2 = 0;//locality�� ������ �ڵ��� ����ð� ����

	float* a = (float*)malloc(sizeof(float) * N * N);//��� a
	float* b = (float*)malloc(sizeof(float) * N * N);//��� b
	float* transpose_b = (float*)malloc(sizeof(float) * N * N);//locality�� ���� b�� transpose���� ��

	float* result1 = (float*)malloc(sizeof(float) * N * N);//locality�� �������� ���� �ڵ��� ����� ����
	float* result2 = (float*)malloc(sizeof(float) * N * N);//locality�� ������ �ڵ��� ����� ����


	init_array(a, b, transpose_b, result1, result2);//��ĵ� �ʱ�ȭ

	//locality�� �������� ���� ���
	CHECK_TIME_START;//�ð� üũ ����
	without_locality(a, b, result1);
	CHECK_TIME_END(compute_time1);//�ð�üũ ����

	//locality�� ������ ���
	CHECK_TIME_START;//�ð� üũ ����
	with_locality(a, transpose_b, result2);
	CHECK_TIME_END(compute_time2);//�ð�üũ ����

	check_results(result1, result2);

	printf("\n\n����ð�\n");
	printf("locality�� ������� ���� �ڵ�: %f ms\n", compute_time1);
	printf("locality�� ����� �ڵ�: %f ms\n", compute_time2);

	

	//�޸� deallocate.
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
			a[i * N + j] = cnt_a++;//a ����� ���ǰ��� ����־� �ʱ�ȭ ����
			b[i * N + j] = cnt_b++;//b ����� ���ǰ��� ����־� �ʱ�ȭ ����
			result1[i * N + j] = 0;//result1 ����� 0���� �ʱ�ȭ����
			result2[i * N + j] = 0;//result2 ����� 0���� �ʱ�ȭ����
		}
	}

	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			transpose_b[i * N + j] = b[j * N + i];//b ����� transpose���� transpose_b�� ��������
}


//�� result�� ���� ���� ������, �� ���� ����� ������ ������ check
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

//�޸��� locality�� ������� ���� �ڵ�
void without_locality(float *a, float *b, float *result)
{
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			for (int k = 0; k < N; k++)
			{
				result[i * N + j] += a[i * N + k] * b[k * N + j];
				//�� iteration���� b�� ������ ����, �������� memory ������ �����ϴ� ���� �ƴ�, ������ �����ϰ� �� 
				//spatial locality�� ������� ����.
			}
		}
	}
}
//�޸��� locality�� ����� �ڵ�
void with_locality(float *a, float *transpose_b, float *result)
{
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			for (int k = 0; k < N; k++)
			{
				result[i * N + j] += a[i * N + k] * transpose_b[j * N + k];
				//transpose_b�� ����Ͽ�, �� iteration���� b�� ������ ��, �������� memory ������ ������ �� ����
				//spatial locality�� ���
			}
		}
	}
}
