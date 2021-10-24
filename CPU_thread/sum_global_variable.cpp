#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <mutex>
#include <chrono>//�ð��� �����ϱ����� ���

using namespace std;
using namespace std::chrono;

#define MAX_THREADS 2048
/* 0���� n-1������ sum�� ���ϴ� �۾��� �����ϰ� �� */


void sum_global_variable(void* vargp);// thread�� ������ ��ƾ

//�����Ǵ� ����
long long global_sum = 0;  //��ü sum�� �����ϴµ� ���Ǵ� ����
long num_per_thread; //thread�� ���� ���� ����
mutex lock_mutex;
long long N = (1 << 30);

int main(int argc, char** argv)
{
    long i;
    long id[MAX_THREADS]; //�� ��° thread������ �����ϴµ� ���Ǵ� ������ �̸� ������� �ڽ��� ������ �˰� ��
    thread threads[MAX_THREADS]; //thread�� �����ϴµ� ���Ǵ� ����
    long thread_num;

    
    //thread�� ������ �ٲپ� ���鼭 test
    for (thread_num = 1; thread_num <= MAX_THREADS; thread_num *= 2)
    {
        //�ʿ��� ������ �ʱ�ȭ
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

        printf("%d���� thread �̿�\n", thread_num);
        //���� �´����� ���θ� üũ
        if (global_sum != (N * (N - 1)) / 2)
        {
            printf("Result is not correct\n");
        }
        else
        {
            printf("Result is correct\n");
        }

        printf("�ɸ��ð� : %d ms\n\n", duration_cast<milliseconds>(end - start).count());
    }
    exit(0);
}
//�� thread�� ������ �Լ�
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
        global_sum += i;//global_sum�̶�� ���������͸� �����ϴ� �ڵ念��, �� ũ��Ƽ�ü����� semaphore�� ������
        lock_mutex.unlock();
    }
}