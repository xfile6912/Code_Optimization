# Code_Optimization Test(C++ Language)
### Environment
  - Processor: Intel(R) Core(TM) i5-9600KF CPU @ 3.70GHz 3.70GHz
  - RAM : 16GB
  - x64 기반 프로세서, Windows10
  - Graphic Card: NVIDIA GeForce GTX1660Ti

### Code Optimization 기법
- Spatial Locality
  - 개념
    - 최근에 접근한 데이터로부터 가까운 곳에 있는 데이터는 다시 이용될 확률이 높아, 특정 데이터를 Caching할 때 근처의 데이터들 역시 Caching하게 됨
  - 개요
    - 행렬의 곱셈(a*b)에서, 일반적으로 코드를 짜게되면 b 행렬에 접근할 때 연속적인 메모리영역을 접근하는 것이 아니므로 상대적으로 cache miss가 많이 발생하게 됨
    - 이 때, b 행렬을 transpose한 행렬을 사용하게 되면 연속적으로 메모리영역을 접근할 수 있어 cache hit가 많이 발생하게됨
    - 따라서 속도가 상대적으로 빠를 것임
  - 코드
    - Locality를 고려한 경우<br>
      ```
      for (int i = 0; i < N; i++)
      {
        for (int j = 0; j < N; j++)
        {
          for (int k = 0; k < N; k++)
          {
            result[i * N + j] += a[i * N + k] * b[k * N + j];
            //매 iteration마다 b를 접근할 때에, 연속적인 memory 영역을 접근하는 것이 아닌, 띄엄띄엄 접근하게 됨 
            //spatial locality를 고려하지 않음.
          }
        }
      }
      ```
    - Locality를 고려하지 않은 경우<br>
      ```
      for (int i = 0; i < N; i++)
      {
        for (int j = 0; j < N; j++)
        {
          for (int k = 0; k < N; k++)
          {
            result[i * N + j] += a[i * N + k] * transpose_b[j * N + k];
            //transpose_b를 사용하여, 매 iteration마다 b를 접근할 때, 연속적인 memory 영역을 접근할 수 있음
            //spatial locality를 고려
          }
        }
      }
      ```
  - 결과<br>
    <img width="400" alt="image" src="https://user-images.githubusercontent.com/57051773/137341608-ad05ca07-058c-4c8b-98ec-307a459f9ab3.png">
    - Locality를 고려한 코드가 더 빠른 것을 확인할 수 있음

- Common Subexpression Elimination
  - 개념
    - 같은 코드를 중복으로 실행하지 않고, 해당 코드를 한번 실행한 값을 저장하고 해당 값을 사용함으로써 최적화하는 것
  - 개요
    - pow(b, c) 코드는 b의 c제곱을 실행하는 함수임
    - 코드 전반에 걸쳐 공통적으로 사용되는 pow(b, c)를 common이라는 변수에 저장하여 이를 사용
    - 따라서 속도가 빠를 것임
  - 코드
    - Common Subexpression Elimination을 하지 않은 경우
      ```
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
      ```
    - Common Subexpression Elimination을 수행한 경우
      ```
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
      ```
  - 결과<br>
    <img width="600" alt="image" src="https://user-images.githubusercontent.com/57051773/137668149-406704c6-1f7e-47d0-b795-147dcc1477e5.png">
    - Common Subexpression Elimination을 수행한 코드가 더 빠른 것을 확인할 수 있음
- Loop Unrolling
  - 개념
    - Loop안의 내용을 펼쳐서 비교와 관련된 연산을 줄여 코드를 최적화하는 것
  - 개요
    - Loop에서는 지정된 범위에 도달했는지의 여부에 대해 검사하기 위해, 값을 비교하고 증가 및 감소 시키는 작업이 진행됨
    - Loop를 펼쳐서 이러한 과정을 줄이게 됨
    - 따라서 속도가 빠를 것임
  - 코드
    - Loop Unrolling를 하지 않은 코드
    ```
    void without_loop_unrolling(float* a, float* result)
    {
        for (int i = 0; i < pow(N, 2); i++)//값을 비교하는 작업에서의 시간을 확실히 비교하기 위해 N*N을 사용한 것이 아닌 pow(N, 2)사용
        {
	   	*result += a[i];
        }
    }
    ```
    - Loop Unrolling를 한 코드의 경우
    ```
    void with_loop_unrolling(float* a, float* result)
    {
        for (int i = 0; i < pow(N, 2); i+=4)//값을 비교하는 작업에서의 시간을 확실히 비교하기 위해 N*N을 사용한 것이 아닌 pow(N, 2)사용
        {
		*result += a[i];
		*result += a[i + 1];
		*result += a[i + 2];
		*result += a[i + 3];
        }
    }
    ```
  - 결과<br>
    <img width="500" alt="image" src="https://user-images.githubusercontent.com/57051773/137896397-8b9eb888-681c-41b3-a80f-5b7e6bb59f2e.png">
    - Loop Unrolling을 수행한 코드가 더 빠른 것을 확인할 수 있음
- Function Inlining
  - 개념
    - 작은 크기의 함수의 경우에는, 함수로 따로 만들지 않고 로직을 그대로 코드에 삽입하여 최적화
  - 개요
    - 함수를 호출하게 되면, return adress를 스택에 저장하고 해당 함수로 jump하게 됨
    - 또한 함수에서 사용되는 일반적인 register를 스택에 백업하고, register 값을 복원하는 과정을 거치면서 시간이 소요되게 된다.
    - 작은 크기의 함수의 경우, 함수로 따로 만들지 않고 로직을 그대로 코드에 삽입하여 시간을 줄이게 됨
    - 따라서 속도가 빠를 것임
  - 코드
    - Function Inlining을 하지 않은 코드의 경우
      ```
      float add_func(float a, float b)
      {
            float ret = a + b;
            return ret;
      }
      void without_function_inlining(float* a, float* b, float* result)
      {
            for (int i = 0; i < pow(N, 2); i++)
            {
                      *result += add_func(a[i], b[i]);
            }
      }
      ```
    - Function Inlining을 한 코드의 경우
      ```
      void with_function_inlining(float* a, float* b, float* result)
      {
            for (int i = 0; i < pow(N, 2); i++)
            {
                     *result += a[i] + b[i];
            }
      }
      ```
  - 결과<br>
    <img width="450" alt="image" src="https://user-images.githubusercontent.com/57051773/138059151-6c8f3314-9be7-4666-9bd1-7479ba3000c5.png">
    - Function Inlining을 수행한 코드가 더 빠른 것을 확인할 수 있음
- Code Motion
  - 개념
    - 기존의 코드를 새로운 위치로 이동함으로써 최적화
  - 개요
    - a + pow(b, c)는 반복문 안에서 지속적인 시간을 소요하지만 따로 값의 변화가 일어나지는 않음
    - a + pow(b, c)를 반복문 바깥으로 옮김으로써 이러한 지속적인 시간소요를 줄이게 됨
    - 따라서 속도가 빠를 것임
  - 코드
    - Code Motion을 하지 않은 코드의 경우
      ```
      void without_code_motion(float a, float b, float c, float* result)
      {
           for (int i = 0; i < N * N; i++)
           {
	          //result를 구할 때에 a+pow(b,c)가 들어가게 되는데 이 값은 루프를 돌면서 변하지 않음
	          *result += (i % 5) + a + pow(b, c);
           }
      }
      ```
    - Code Motion을 한 코드의 경우
      ```
      void with_code_motion(float a, float b, float c, float* result)
      {
           //따라서 code_motion에서는 a + pow(b,c)를 for문 밖으로 옮겨줌
           float temp = a + pow(b, c);
           for (int i = 0; i < N * N; i++)
           {
                   *result += (i % 5) + temp;
           }
      }     
      ```
  - 결과<br>
    <img width="400" alt="image" src="https://user-images.githubusercontent.com/57051773/138059197-54337595-e9ac-422f-b153-2aa8ed5e5f72.png">
    - Code Motion을 수행한 코드가 더 빠른 것을 확인할 수 있음
- Instruction Scheduling
### CPU(Thread Num)
##### 1부터 N-1까지를 더하는 코드를 각 경우에 맞게 작성하여 시간을 측정
- Global Variable만을 활용하는 경우
  - 개요
    - 모든 thread에서 공유하는 global_sum이라는 공유변수를 이용
    - 각 thread에서는 자신이 맡은 범위까지의 덧셈을 바로바로 global_sum에 반영
    - 해당 과정에서 각 thread가 global_sum에 접근할 때는 올바른 값의 도출을 위해 semaphore로 감싸주게 됨
  - 코드
    ```
    //각 thread가 실행할 함수
    void sum_global_variable(void* vargp)
    {
        //num_per_thread의 값은 read만 하기 때문에 따로 semaphore로 감싸줄 필요가 없음
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
    ```
  - 결과<br>
    <img width="500" alt="image" src="https://user-images.githubusercontent.com/57051773/138621033-10a63fc8-adf7-4807-887b-8fb7ebc40850.png">
    - thread의 수를 늘림에도 불구하고, 오히려 느려지는 것을 확인할 수 있음
    - 하나의 공유 데이터에 접근하기 때문에, thread를 사용하더라도 semaphore를 통해 순차적으로 접근하는 것과 같음
    - 또한 매번 더할 때마다 semaphore에 의한 overhead가 발생하기 때문임
- Global Array를 사용해 Mutex Overhead를 줄인 경우
  - 개요
    - 메인 thread에서는 global_sum이라는 변수를 이용
    - 모든 thread에서는 global_array라는 공유변수 배열을 이용
    - 각 thread는 자신이 맡은 범위까지의 덧셈을 수행하여 자신이 담당하는 global_array의 해당 index에 저장
    - 마지막으로 메인 thread가 이러한 global_array의 값들을 취합하여 global_sum에 최종 값을 저장
  - 코드
    ```
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
        //num_per_thread의 값은 read만 하기 때문에 따로 semaphore로 감싸줄 필요가 없음
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
    ```
  - 결과<br>
    <img width="500" alt="image" src="https://user-images.githubusercontent.com/57051773/138621083-7291746f-4549-476c-a986-dafd8b3d9879.png">
    - Thread의 수가 적절히 많아질 수록 속도가 빨라지는 경향을 확인할 수 있음
      - 따라서 프로세서의 Core 수에 따라 적절히 Thread의 수를 정해주는 것이 필요
    - Global Variable에 비해 속도가 빠른 것을 확인 할 수 있음
      - 각 Thread는 Global Array의 자신의 Index에 해당하는 부분만 접근하기 때문에 Semaphore를 사용하지 않음
      - 이를 통해 Semaphore에 따른 Overhead가 줄어들었기 때문에 속도가 빨라짐
- Thread내의 Local Variable을 사용해 Register를 사용하도록 한 경우
  - 개요
    - 메인 thread에서는 global_sum이라는 변수를 이용
    - 모든 thread에서는 global_array라는 공유변수 배열을 이용
    - 모든 thread에서는 local_sum이라는 지역변수를 이용
    - 각 thread는 자신이 맡은 범위까지의 덧셈의 과정을 자신의 지역변수인 local_sum을 이용해 수행
    - 각 thread는 최종적으로 local_sum의 값을 global_array의 자신이 담당하는 index의 위치에 대입
    - 마지막으로 메인 thread가 이러한 global_array의 값들을 취합하여 global_sum에 최종 값을 저장
  - 코드
    ```
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
    ```
  - 결과<br>
    <img width="500" alt="image" src="https://user-images.githubusercontent.com/57051773/138621714-6376341c-745d-4336-8f5f-9b9f8dad5f92.png">
    - Thread의 수가 적절히 많아질 수록 속도가 빨라지는 경향을 확인할 수 있음
      - 따라서 프로세서의 Core 수에 따라 적절히 Thread의 수를 정해주는 것이 필요
    - Global Array에 비해 속도가 빠른 것을 확인 할 수 있음
      - 각 thread에서 계산시 지역변수(local_sum)을 이용하게 되면 Compiler에 의해 Register를 사용하도록 하기 때문에 Memory Reference가 줄어들게 되고 속도가 더 빨라짐
### CPU vs GPU(CUDA)
- 
### GPU(Non Shared Memory) vs GPU(Shared Memory)
- 
