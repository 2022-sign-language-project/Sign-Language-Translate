## 개발환경
anaconda virtual env (test)
python 3.8

```
# in cmd
conda activate test
jupyter notebook
# or
code
```

### pip freeze.txt 
파일을 통해 모든 모듈 확인가능

### 현재 그래픽
Intel(R) Iris(R) Xe Graphics

### 결과
Accuracy 가 많이 불안정하다, 할때마다 결과가 많이 다르다.


### 중요한 코드 수정 부분
```
# Path for exproted data, numpy arrays
DATA_PATH =os.path.join('MP_DATA')
# Actions that we try to detect
actions = np.array(['Hongkong','Poland','Turkey','China','France','Brazil','Germany','Thai','US','Mexico'])
# Thirty videos worth of data
no_sequences = 10
# Videos are going to be 30 frames in length
sequence_length = 30
```

