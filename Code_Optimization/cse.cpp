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
void without_cse(float a, float b, float c, float* result);
void with_cse(float a, float b, float c, float* result);

//common subexpression elimination�� cse��� ��Ī��
//cse�� ���� ���� �ڵ�� cse�� �� �ڵ��� ����ð��� ��
int main(void)
{
	float compute_time1 = 0;//cse�� ���� ���� �ڵ��� ����ð� ����
	float compute_time2 = 0;//cse�� �� �ڵ��� ����ð� ����

	//�񱳸� ���� ���Ǵ� ������ �ʱ�ȭ
	float a = 1.1;
	float b = 2.2;
	float c = 3.3;
	float result1 = 0.0;//cse�� ���� ���� �ڵ��� ����� ����
	float result2 = 0.0;//cse�� �� �ڵ��� ����� ����

	//cse�� ���� ���� �ڵ��� ���
	CHECK_TIME_START;//�ð� üũ ����
	without_cse(a, b, c, &result1);
	CHECK_TIME_END(compute_time1);//�ð�üũ ����

	//cse�� �� �ڵ��ǰ��
	CHECK_TIME_START;//�ð� üũ ����
	with_cse(a, b, c, &result2);
	CHECK_TIME_END(compute_time2);//�ð�üũ ����

	check_results(result1, result2);

	printf("\n\n����ð�\n");
	printf("common subexpression elimination�� ���� ���� �ڵ�: %f ms\n", compute_time1);
	printf("common subexpression elimination�� �� �ڵ�: %f ms\n", compute_time2);


}


//�� result�� ���� ���� ������, �� ���� ����� ������ ������ check
void check_results(float result1, float result2)
{
	if (result1 == result2)
		printf("Results are same\n");
	else
		printf("Results are not same\n");
}

//common subexpression elimination�� ���� ���� �ڵ�
void without_cse(float a, float b, float c, float* result)
{
	for (int i = 0; i < N * N; i++)
	{
		//result�� ���ϴ� ���꿡 pow(b, c)�� ������ ��
		*result += pow(b, c) + pow(b, c) * a;
		*result -= pow(b, c) - pow(b, c) + b;
		b++; c++;
	}
}
//common subexpression elimination�� �� �ڵ�
void with_cse(float a, float b, float c, float* result)
{
	for (int i = 0; i < N * N; i++)
	{
		//���������� ���� pow(b,c)�� common�̶�� ������ �������ְ�
		//�̸� result�� ��꿡 ���
		double common = pow(b, c);
		*result += common + common * a;
		*result -= common - common + b;
		b++; c++;
	}
}
