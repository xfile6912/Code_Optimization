#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <chrono>//시간을 측정하기위해 사용

using namespace std;
using namespace std::chrono;

//이전의 CPU에서 가장 시간이 최적화 되었었던 32개의 THREAD를 사용
#define MAX_THREADS 32
#define MAX_N_ELEMENTS (1<<27)

//Array의 Sum 진행

void init_arrays();//A, B, C_RESULT Array를 초기화하는 함수
void sum_array(void* vargp);// thread가 실행할 루틴

long num_per_thread; //thread당 맡은 수의 개수
float *C_RESULT;//두 array의 합이 저장되는 배열
float* A;
float* B;

int main(void)
{
    long i;
    long id[MAX_THREADS]; //몇 번째 thread인지를 저장하는데 사용되는 변수로 이를 기반으로 자신의 범위를 알게 됨
    thread threads[MAX_THREADS]; //thread를 저장하는데 사용되는 변수

    //필요한 정보들 초기화
    num_per_thread = MAX_N_ELEMENTS / MAX_THREADS;
    init_arrays();

    //CPU 시간측정
    auto start = high_resolution_clock::now();

    for (i = 0; i < MAX_THREADS; i++) {
        id[i] = i;
        threads[i] = thread(sum_array, &id[i]);
    }

    for (i = 0; i < MAX_THREADS; i++)
    {
        threads[i].join();
    }
    //CPU시간측정 완료
    auto end = high_resolution_clock::now();

    //결과 출력
    printf("CPU_RESULT[10] = %f\n", C_RESULT[10]);
    printf("걸린시간 : %d ms\n\n", duration_cast<milliseconds>(end - start).count());

    free(A);
    free(B);
    free(C_RESULT);
    exit(0);
}
//A, B, C_RESULT Array를 초기화하는 함수
void init_arrays()
{
    int i;
    A = (float*)malloc(sizeof(float) * MAX_N_ELEMENTS);
    B = (float*)malloc(sizeof(float) * MAX_N_ELEMENTS);
    C_RESULT = (float*)malloc(sizeof(float) * MAX_N_ELEMENTS);

    //array의 각 원소에 float값을 생성해 줌
    for (i = 0; i < MAX_N_ELEMENTS; i++) {
        A[i] = i + 3.14f;
        B[i] = i + 3.14f;
    }
}
//각 thread가 실행할 함수
void sum_array(void* vargp)
{
    long id = (*(long*)vargp);
    long start = id * num_per_thread;
    long end = start + num_per_thread;

    if (end > MAX_N_ELEMENTS)
        end = MAX_N_ELEMENTS;

    long i;
    for (i = start; i < end; i++)
    {
        C_RESULT[i] = A[i] + B[i];//array에서 자신이 맡은 영역의 array의 합을 구해주도록 함
    }

}