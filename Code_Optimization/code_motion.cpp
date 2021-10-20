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
void without_code_motion(float a, float b, float c, float* result);
void with_code_motion(float a, float b, float c, float* result);

//code_motion�� ���� ���� �ڵ�� code_motion�� �� �ڵ��� ����ð��� ��
int main(void)
{
	float compute_time1 = 0;//code_motion�� ���� ���� �ڵ��� ����ð� ����
	float compute_time2 = 0;//code_motion�� �� �ڵ��� ����ð� ����

	//�񱳸� ���� ���Ǵ� ������ �ʱ�ȭ
	float a = 1.1;
	float b = 2.2;
	float c = 3.3;
	float result1 = 0.0;//code_motion�� ���� ���� �ڵ��� ����� ����
	float result2 = 0.0;//code_motion�� �� �ڵ��� ����� ����

	//code_motion�� ���� ���� �ڵ��� ���
	CHECK_TIME_START;//�ð� üũ ����
	without_code_motion(a, b, c, &result1);
	CHECK_TIME_END(compute_time1);//�ð�üũ ����

	//code_motion�� �� �ڵ��ǰ��
	CHECK_TIME_START;//�ð� üũ ����
	with_code_motion(a, b, c, &result2);
	CHECK_TIME_END(compute_time2);//�ð�üũ ����

	check_results(result1, result2);

	printf("\n\n����ð�\n");
	printf("code motion�� ���� ���� �ڵ�: %f ms\n", compute_time1);
	printf("code motion�� �� �ڵ�: %f ms\n", compute_time2);


}


//�� result�� ���� ���� ������, �� ���� ����� ������ ������ check
void check_results(float result1, float result2)
{
	if (result1 == result2)
		printf("Results are same\n");
	else
		printf("Results are not same\n");
}

//code motion�� ���� ���� �ڵ�
void without_code_motion(float a, float b, float c, float* result)
{
	for (int i = 0; i < N * N; i++)
	{
		//result�� ���� ���� a+pow(b,c)�� ���� �Ǵµ� �� ���� ������ ���鼭 ������ ����
		*result += (i % 5) + a + pow(b, c);
	}
}
//code motion�� �� �ڵ�
void with_code_motion(float a, float b, float c, float* result)
{
	//���� code_motion������ a + pow(b,c)�� for�� ������ �Ű���
	float temp = a + pow(b, c);
	for (int i = 0; i < N * N; i++)
	{
		*result += (i % 5) + temp;
	}
}
