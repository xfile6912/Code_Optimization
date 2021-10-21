#include "csapp.h"
#include "time.h"

#define MAX_THREADS 1024
#define N 1<<31
/* 0부터 n-1까지의 sum을 구하는 작업을 진행하게 됨 */

void *sum_global_variable(void *vargp);// thread가 실행할 루틴

//공유되는 변수
long global_sum=0;  //전체 sum을 저장하는데 사용되는 변수
long num_per_thread; //thread당 맡은 수의 개수
long n;
sem_t mutex;    //global_sum 공유 변수를 보호하기 위한 뮤텍스

int main(int argc, char **argv)
{
    long i;
    long id[MAX_THREADS]; //몇 번째 thread인지를 저장하는데 사용되는 변수로 이를 기반으로 자신의 범위를 알게 됨
    pthread_t threads[MAX_THREADS]; //thread를 저장하는데 사용되는 변수
    long thread_num;

    clock_t start, end;

    //필요한 정보들 초기화
    thread_num = atoi(argv[1]);
    num_per_thread = N/thread_num;
    sem_init(&mutex, 0, 1);//mutex 초기화
    
    start = clock();

    for(i=0; i<thread_num; i++){
        id[i] = i;
        Pthread_create(&threads[i], NULL, sum_global_variable, &id[i]);//thread를 생성해줌
    }

    for(i=0; i<thread_num; i++)
    {
        Pthread_join(threads[i], NULL);//모든 thread들이 끝날 때까지 기다림
    }

    end = clock();
     
    //답이 맞는지의 여부를 체크
    if(global_sum != (N * (N-1))/2)
    {
        printf("Result is not correct\n");
    }
    else
    {
        printf("Result is correct\n");
    }

    printf("걸린시간 : %f", (double)end-start);

    exit(0);
}
//각 thread가 실행할 함수
void *sum_global_variable(void *vargp)
{
    //num_per_thread의 값은 read만 하기 때문에 따로 semaphore로 감싸줄 필요가 없음
    long id=(*(long*)vargp);
    long start = id*num_per_thread;
    long end =start + num_per_thread;
    if(end>N)
        end=N;
    
    long i;
    for(i=start; i<end; i++)
    {
        P(&mutex);
        global_sum+=i;//global_sum이라는 공유데이터를 변경하는 코드영역, 즉 크리티컬섹션을 semaphore로 감싸줌
        V(&mutex);
    }
    return NULL;
}