#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <chrono>//시간을 측정하기위해 사용

using namespace std;
using namespace std::chrono;

#define MAX_THREADS 2048
/* 0부터 n-1까지의 sum을 구하는 작업을 진행하게 됨 */


void sum_global_array(void* vargp);// thread가 실행할 루틴
void total_sum(int thread_num);

//공유되는 변수
long long global_sum = 0;  //전체 sum을 저장하는데 사용되는 변수
long num_per_thread; //thread당 맡은 수의 개수
long long global_array[MAX_THREADS];//각 thread가 자신이 맡은 영역에 대한 합을 저장하는 배열
long long N = (1 << 30);

int main(int argc, char** argv)
{
    long i;
    long id[MAX_THREADS]; //몇 번째 thread인지를 저장하는데 사용되는 변수로 이를 기반으로 자신의 범위를 알게 됨
    thread threads[MAX_THREADS]; //thread를 저장하는데 사용되는 변수
    long thread_num;


    //thread의 개수를 바꾸어 가면서 test
    for (thread_num=1; thread_num <= MAX_THREADS; thread_num *= 2)
    {

        //필요한 정보들 초기화
        global_sum = 0;
        memset(global_array, 0, sizeof(global_array));
        num_per_thread = N / thread_num;
        auto start = high_resolution_clock::now();

        for (i = 0; i < thread_num; i++) {
            id[i] = i;
            threads[i] = thread(sum_global_array, &id[i]);
        }

        for (i = 0; i < thread_num; i++)
        {
            threads[i].join();
        }

        //최종적으로 global_array에 있는 내용들을 취합해서 global_sum에 저장해주는 과정 수행
        total_sum(thread_num);

        auto end = high_resolution_clock::now();

        printf("%d개의 thread 이용\n", thread_num);
        //답이 맞는지의 여부를 체크
        if (global_sum != (N * (N - 1)) / 2)
        {
            printf("Result is not correct\n");
        }
        else
        {
            printf("Result is correct\n");
        }
        printf("걸린시간 : %d ms\n\n", duration_cast<milliseconds>(end - start).count());

    }
    exit(0);
}


//메인 thread에서 최종적으로 global_array의 내용들을 취합해서 sum을 하는 과정
void total_sum(int thread_num)
{
    long i;
    for (i = 0; i < thread_num; i++)
        global_sum += global_array[i];
}

//각 thread가 실행할 함수
void sum_global_array(void* vargp)
{
    long id = (*(long*)vargp);
    long start = id * num_per_thread;
    long end = start + num_per_thread;
    if (end > N)
        end = N;

    long i;
    for (i = start; i < end; i++)
    {
        global_array[id] += i;//global_array의 자신이 담당하는 index에, 자신이 맡은 영역의 sum을 저장해줌
    }
}