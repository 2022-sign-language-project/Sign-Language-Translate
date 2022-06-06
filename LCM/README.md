# Skeleton code

```
create_dataset.py:
cv window에 goolge mediapipe holistic을 이용하여 관절부위 등에 keypoints를 찍어줌.
한 단어당 수어를 30번 수행후 내용을 .npy로 저장
```

```
train.py:
수어 데이터를 preprocessing하고 RNN방식으로 학습가능(# 현재는 **CNN** 이용)
```

```
test.py:
실행시 cv window에 keypoints가 찍히고 단어 수행시 왼쪽에 단어의 예측 정도가 보임
위쪽엔 threshold값을 넘은 단어가 뜨게 되있음
```

# CNN

```
train_cnn.py:
CNN 이용
skeleton code 디렉토리 내의 train.py와는 다르게 CNN이용하기 때문에 데이터 전처리시 dimension을 하나 늘려줌
Conv2D, MaxPooling2D, Dropout, Flatten, Dense가 이용됨
```

```
test.py:
실행시 cv window에 keypoints가 찍히고 단어 수행시 왼쪽에 단어의 예측 정도가 보임
위쪽엔 threshold값을 넘은 단어가 뜨게 되있음
많은 단어 수용을 위해 skeleton code 디렉토리 내의 test.py와 다르게 왼쪽 단어의 크기를 줄임
```

# model

```
20220606_90.h5:
시나리오에 따라 선택한 단어 12개에 대한 팀원 3명의 데이터를 모두 모가 학습 시킨 모델
(동생에게 부탁하여 수어 수행한 결과 성공적 => 다른 인원이 해도 인식률이 나쁘지 않다고 확인)
```
