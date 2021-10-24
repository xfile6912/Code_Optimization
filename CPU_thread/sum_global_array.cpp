#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <chrono>//�ð��� �����ϱ����� ���

using namespace std;
using namespace std::chrono;

#define MAX_THREADS 2048
/* 0���� n-1������ sum�� ���ϴ� �۾��� �����ϰ� �� */


void sum_global_array(void* vargp);// thread�� ������ ��ƾ
void total_sum(int thread_num);

//�����Ǵ� ����
long long global_sum = 0;  //��ü sum�� �����ϴµ� ���Ǵ� ����
long num_per_thread; //thread�� ���� ���� ����
long long global_array[MAX_THREADS];//�� thread�� �ڽ��� ���� ������ ���� ���� �����ϴ� �迭
long long N = (1 << 30);

int main(int argc, char** argv)
{
    long i;
    long id[MAX_THREADS]; //�� ��° thread������ �����ϴµ� ���Ǵ� ������ �̸� ������� �ڽ��� ������ �˰� ��
    thread threads[MAX_THREADS]; //thread�� �����ϴµ� ���Ǵ� ����
    long thread_num;


    //thread�� ������ �ٲپ� ���鼭 test
    for (thread_num=1; thread_num <= MAX_THREADS; thread_num *= 2)
    {

        //�ʿ��� ������ �ʱ�ȭ
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

        //���������� global_array�� �ִ� ������� �����ؼ� global_sum�� �������ִ� ���� ����
        total_sum(thread_num);

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


//���� thread���� ���������� global_array�� ������� �����ؼ� sum�� �ϴ� ����
void total_sum(int thread_num)
{
    long i;
    for (i = 0; i < thread_num; i++)
        global_sum += global_array[i];
}

//�� thread�� ������ �Լ�
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
        global_array[id] += i;//global_array�� �ڽ��� ����ϴ� index��, �ڽ��� ���� ������ sum�� ��������
    }
}