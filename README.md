# Code_Optimization Test(C Language)
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
  - 결과
    - Loop Unrolling을 수행한 코드가 더 빠른 것을 확인할 수 있음
- Function Inlining
- Code Motion
- Instruction Scheduling
### CPU(Thread num)
- 
### CPU vs GPU(CUDA)
- 
### GPU(Non Shared Memory) vs GPU(Shared Memory)
- 
