#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <mutex>
#include <chrono>//시간을 측정하기위해 사용

using namespace std;
using namespace std::chrono;

#define MAX_THREADS 2048
/* 0부터 n-1까지의 sum을 구하는 작업을 진행하게 됨 */


void sum_global_variable(void* vargp);// thread가 실행할 루틴

//공유되는 변수
long long global_sum = 0;  //전체 sum을 저장하는데 사용되는 변수
long num_per_thread; //thread당 맡은 수의 개수
mutex lock_mutex;
long long N = (1 << 30);

int main(int argc, char** argv)
{
    long i;
    long id[MAX_THREADS]; //몇 번째 thread인지를 저장하는데 사용되는 변수로 이를 기반으로 자신의 범위를 알게 됨
    thread threads[MAX_THREADS]; //thread를 저장하는데 사용되는 변수
    long thread_num;

    
    //thread의 개수를 바꾸어 가면서 test
    for (thread_num = 1; thread_num <= MAX_THREADS; thread_num *= 2)
    {
        //필요한 정보들 초기화
        global_sum = 0;
        num_per_thread = N / thread_num;

        auto start = high_resolution_clock::now();

        for (i = 0; i < thread_num; i++) {
            id[i] = i;
            threads[i] = thread(sum_global_variable, &id[i]);
        }

        for (i = 0; i < thread_num; i++)
        {
            threads[i].join();
        }

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
//각 thread가 실행할 함수
void sum_global_variable(void* vargp)
{
    long id = (*(long*)vargp);
    long start = id * num_per_thread;
    long end = start + num_per_thread;
    if (end > N)
        end = N;

    long i;
    for (i = start; i < end; i++)
    {
        lock_mutex.lock();
        global_sum += i;//global_sum이라는 공유데이터를 변경하는 코드영역, 즉 크리티컬섹션을 semaphore로 감싸줌
        lock_mutex.unlock();
    }
}