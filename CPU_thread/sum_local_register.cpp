#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <chrono>//시간을 측정하기위해 사용

using namespace std;
using namespace std::chrono;

#define MAX_THREADS 2048
/* 0부터 n-1까지의 sum을 구하는 작업을 진행하게 됨 */


void sum_local_register(void* vargp);// thread가 실행할 루틴
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
    for (thread_num = 1; thread_num <= MAX_THREADS; thread_num *= 2)
    {

        //필요한 정보들 초기화
        global_sum = 0;
        memset(global_array, 0, sizeof(global_array));
        num_per_thread = N / thread_num;
        auto start = high_resolution_clock::now();

        for (i = 0; i < thread_num; i++) {
            id[i] = i;
            threads[i] = thread(sum_local_register, &id[i]);
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
void sum_local_register(void* vargp)
{
    long id = (*(long*)vargp);
    long start = id * num_per_thread;
    long end = start + num_per_thread;
    //thread에 local변수를 만들어서 이를 이용하도록 하여
    //메인 메모리에 대한 접근을 줄이고
    //(컴파일러가 최적화 하면서 local변수는 register를 이용하도록 하기 때문에)
    //메인 메모리 영역에 있는 global_array는 
    //한번만 접근하게 해줌으로써 속도를 높임
    long long local_sum = 0;
    if (end > N)
        end = N;

    long i;
    for (i = start; i < end; i++)
    {
        local_sum += i;//local_sum을 이용해 자신이 맡은 영역의 sum을 계산해주도록 함
    }

    //global array는 각 thread마다 한번만 접근함
    global_array[id] = local_sum;
}