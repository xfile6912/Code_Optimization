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

//common subexpression elimination을 cse라고 지칭함
//cse를 하지 않은 코드와 cse를 한 코드의 수행시간을 비교
int main(void)
{
	float compute_time1 = 0;//cse를 하지 않은 코드의 수행시간 저장
	float compute_time2 = 0;//cse를 한 코드의 수행시간 저장

	//비교를 위해 사용되는 변수들 초기화
	float a = 1.1;
	float b = 2.2;
	float c = 3.3;
	float result1 = 0.0;//cse를 하지 않은 코드의 결과값 저장
	float result2 = 0.0;//cse를 한 코드의 결과값 저장

	//cse를 하지 않은 코드의 경우
	CHECK_TIME_START;//시간 체크 시작
	without_cse(a, b, c, &result1);
	CHECK_TIME_END(compute_time1);//시간체크 종료

	//cse를 한 코드의경우
	CHECK_TIME_START;//시간 체크 시작
	with_cse(a, b, c, &result2);
	CHECK_TIME_END(compute_time2);//시간체크 종료

	check_results(result1, result2);

	printf("\n\n수행시간\n");
	printf("common subexpression elimination을 하지 않은 코드: %f ms\n", compute_time1);
	printf("common subexpression elimination을 한 코드: %f ms\n", compute_time2);


}


//두 result의 값이 서로 같은지, 즉 같은 결과를 도출해 내는지 check
void check_results(float result1, float result2)
{
	if (result1 == result2)
		printf("Results are same\n");
	else
		printf("Results are not same\n");
}

//common subexpression elimination을 하지 않은 코드
void without_cse(float a, float b, float c, float* result)
{
	for (int i = 0; i < N * N; i++)
	{
		//result를 구하는 연산에 pow(b, c)가 여러번 들어감
		*result += pow(b, c) + pow(b, c) * a;
		*result -= pow(b, c) - pow(b, c) + b;
		b++; c++;
	}
}
//common subexpression elimination을 한 코드
void with_cse(float a, float b, float c, float* result)
{
	for (int i = 0; i < N * N; i++)
	{
		//공통적으로 들어가는 pow(b,c)를 common이라는 변수에 저장해주고
		//이를 result값 계산에 사용
		double common = pow(b, c);
		*result += common + common * a;
		*result -= common - common + b;
		b++; c++;
	}
}
