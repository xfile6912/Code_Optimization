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

//code_motion를 하지 않은 코드와 code_motion를 한 코드의 수행시간을 비교
int main(void)
{
	float compute_time1 = 0;//code_motion를 하지 않은 코드의 수행시간 저장
	float compute_time2 = 0;//code_motion를 한 코드의 수행시간 저장

	//비교를 위해 사용되는 변수들 초기화
	float a = 1.1;
	float b = 2.2;
	float c = 3.3;
	float result1 = 0.0;//code_motion를 하지 않은 코드의 결과값 저장
	float result2 = 0.0;//code_motion를 한 코드의 결과값 저장

	//code_motion를 하지 않은 코드의 경우
	CHECK_TIME_START;//시간 체크 시작
	without_code_motion(a, b, c, &result1);
	CHECK_TIME_END(compute_time1);//시간체크 종료

	//code_motion를 한 코드의경우
	CHECK_TIME_START;//시간 체크 시작
	with_code_motion(a, b, c, &result2);
	CHECK_TIME_END(compute_time2);//시간체크 종료

	check_results(result1, result2);

	printf("\n\n수행시간\n");
	printf("code motion을 하지 않은 코드: %f ms\n", compute_time1);
	printf("code motion을 한 코드: %f ms\n", compute_time2);


}


//두 result의 값이 서로 같은지, 즉 같은 결과를 도출해 내는지 check
void check_results(float result1, float result2)
{
	if (result1 == result2)
		printf("Results are same\n");
	else
		printf("Results are not same\n");
}

//code motion을 하지 않은 코드
void without_code_motion(float a, float b, float c, float* result)
{
	for (int i = 0; i < N * N; i++)
	{
		//result를 구할 때에 a+pow(b,c)가 들어가게 되는데 이 값은 루프를 돌면서 변하지 않음
		*result += (i % 5) + a + pow(b, c);
	}
}
//code motion을 한 코드
void with_code_motion(float a, float b, float c, float* result)
{
	//따라서 code_motion에서는 a + pow(b,c)를 for문 밖으로 옮겨줌
	float temp = a + pow(b, c);
	for (int i = 0; i < N * N; i++)
	{
		*result += (i % 5) + temp;
	}
}
