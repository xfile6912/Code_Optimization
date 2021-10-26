#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <chrono>//�ð��� �����ϱ����� ���

using namespace std;
using namespace std::chrono;

//������ CPU���� ���� �ð��� ����ȭ �Ǿ����� 32���� THREAD�� ���
#define MAX_THREADS 32
#define MAX_N_ELEMENTS (1<<27)

//Array�� Sum ����

void init_arrays();//A, B, C_RESULT Array�� �ʱ�ȭ�ϴ� �Լ�
void sum_array(void* vargp);// thread�� ������ ��ƾ

long num_per_thread; //thread�� ���� ���� ����
float *C_RESULT;//�� array�� ���� ����Ǵ� �迭
float* A;
float* B;

int main(void)
{
    long i;
    long id[MAX_THREADS]; //�� ��° thread������ �����ϴµ� ���Ǵ� ������ �̸� ������� �ڽ��� ������ �˰� ��
    thread threads[MAX_THREADS]; //thread�� �����ϴµ� ���Ǵ� ����

    //�ʿ��� ������ �ʱ�ȭ
    num_per_thread = MAX_N_ELEMENTS / MAX_THREADS;
    init_arrays();

    //CPU �ð�����
    auto start = high_resolution_clock::now();

    for (i = 0; i < MAX_THREADS; i++) {
        id[i] = i;
        threads[i] = thread(sum_array, &id[i]);
    }

    for (i = 0; i < MAX_THREADS; i++)
    {
        threads[i].join();
    }
    //CPU�ð����� �Ϸ�
    auto end = high_resolution_clock::now();

    //��� ���
    printf("CPU_RESULT[10] = %f\n", C_RESULT[10]);
    printf("�ɸ��ð� : %d ms\n\n", duration_cast<milliseconds>(end - start).count());

    free(A);
    free(B);
    free(C_RESULT);
    exit(0);
}
//A, B, C_RESULT Array�� �ʱ�ȭ�ϴ� �Լ�
void init_arrays()
{
    int i;
    A = (float*)malloc(sizeof(float) * MAX_N_ELEMENTS);
    B = (float*)malloc(sizeof(float) * MAX_N_ELEMENTS);
    C_RESULT = (float*)malloc(sizeof(float) * MAX_N_ELEMENTS);

    //array�� �� ���ҿ� float���� ������ ��
    for (i = 0; i < MAX_N_ELEMENTS; i++) {
        A[i] = i + 3.14f;
        B[i] = i + 3.14f;
    }
}
//�� thread�� ������ �Լ�
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
        C_RESULT[i] = A[i] + B[i];//array���� �ڽ��� ���� ������ array�� ���� �����ֵ��� ��
    }

}