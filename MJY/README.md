# errors
- 모델을 CNN layer로 바뀐 뒤, test_cnn.py 에서 다음과 같은 에러가 발생
```
Traceback (most recent call last):
  File "test_checking.py", line 157, in <module>
    res = model.predict(np.expand_dims(sequence, axis=0))[0]
  File "C:\Users\test\anaconda3\envs\test\lib\site-packages\tensorflow\python\keras\engine\training.py", line 1629, in predict
    tmp_batch_outputs = self.predict_function(iterator)
  File "C:\Users\test\anaconda3\envs\test\lib\site-packages\tensorflow\python\eager\def_function.py", line 828, in __call__
    result = self._call(*args, **kwds)
  File "C:\Users\test\anaconda3\envs\test\lib\site-packages\tensorflow\python\eager\def_function.py", line 871, in _call
    self._initialize(args, kwds, add_initializers_to=initializers)
  File "C:\Users\test\anaconda3\envs\test\lib\site-packages\tensorflow\python\eager\def_function.py", line 725, in _initialize
    self._stateful_fn._get_concrete_function_internal_garbage_collected(  # pylint: disable=protected-access
  File "C:\Users\test\anaconda3\envs\test\lib\site-packages\tensorflow\python\eager\function.py", line 2969, in _get_concrete_function_internal_garbage_collected
    graph_function, _ = self._maybe_define_function(args, kwargs)
  File "C:\Users\test\anaconda3\envs\test\lib\site-packages\tensorflow\python\eager\function.py", line 3361, in _maybe_define_function
    graph_function = self._create_graph_function(args, kwargs)
  File "C:\Users\test\anaconda3\envs\test\lib\site-packages\tensorflow\python\eager\function.py", line 3196, in _create_graph_function
    func_graph_module.func_graph_from_py_func(
  File "C:\Users\test\anaconda3\envs\test\lib\site-packages\tensorflow\python\framework\func_graph.py", line 990, in func_graph_from_py_func
    func_outputs = python_func(*func_args, **func_kwargs)
  File "C:\Users\test\anaconda3\envs\test\lib\site-packages\tensorflow\python\eager\def_function.py", line 634, in wrapped_fn
    out = weak_wrapped_fn().__wrapped__(*args, **kwds)
  File "C:\Users\test\anaconda3\envs\test\lib\site-packages\tensorflow\python\framework\func_graph.py", line 977, in wrapper
    raise e.ag_error_metadata.to_exception(e)
ValueError: in user code:

    C:\Users\test\anaconda3\envs\test\lib\site-packages\tensorflow\python\keras\engine\training.py:1478 predict_function  *
        return step_function(self, iterator)
    C:\Users\test\anaconda3\envs\test\lib\site-packages\tensorflow\python\keras\engine\training.py:1468 step_function  **
        outputs = model.distribute_strategy.run(run_step, args=(data,))
    C:\Users\test\anaconda3\envs\test\lib\site-packages\tensorflow\python\distribute\distribute_lib.py:1259 run
        return self._extended.call_for_each_replica(fn, args=args, kwargs=kwargs)
    C:\Users\test\anaconda3\envs\test\lib\site-packages\tensorflow\python\distribute\distribute_lib.py:2730 call_for_each_replica
        return self._call_for_each_replica(fn, args, kwargs)
    C:\Users\test\anaconda3\envs\test\lib\site-packages\tensorflow\python\distribute\distribute_lib.py:3417 _call_for_each_replica
        return fn(*args, **kwargs)
    C:\Users\test\anaconda3\envs\test\lib\site-packages\tensorflow\python\keras\engine\training.py:1461 run_step  **
        outputs = model.predict_step(data)
    C:\Users\test\anaconda3\envs\test\lib\site-packages\tensorflow\python\keras\engine\training.py:1434 predict_step
        return self(x, training=False)
    C:\Users\test\anaconda3\envs\test\lib\site-packages\tensorflow\python\keras\engine\base_layer.py:998 __call__
        input_spec.assert_input_compatibility(self.input_spec, inputs, self.name)
    C:\Users\test\anaconda3\envs\test\lib\site-packages\tensorflow\python\keras\engine\input_spec.py:234 assert_input_compatibility
        raise ValueError('Input ' + str(input_index) + ' of layer ' +

    ValueError: Input 0 of layer sequential is incompatible with the layer: : expected min_ndim=4, found ndim=3. Full shape received: (None, 30, 1662)
```

sequence의 차원은 3차원인데 원하는 것은 4차원이라서 문제가 생겼고, 이 url을 통해 해결
https://www.pythonfixing.com/2022/02/fixed-valueerror-input-0-of-layer.html 

```
# 원래코드
        # 2. Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])
            predictions.append(np.argmax(res))

# 바뀐코드
        # 2. Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]
        sequence_np = np.expand_dims(sequence, axis=0)
        sequence_np = sequence_np.reshape(sequence_np.shape + (1,))

        if len(sequence) == 30:
            res = model.predict(sequence_np)[0]
            print(actions[np.argmax(res)])
            predictions.append(np.argmax(res))
```

# colab
내가 가진 노트북으로 Nvidia GPU을 사용할 수 없고, colab을 통해 GPU 문제를 해결할 수 있었다.
앞으로 colab을 통해 모델을 생성할 수 있을 것 같다. 아직 파일 업로드 중이라 실행해보지 못 함.
여기서, 다른 문제가 생겼다.
test_cnn.py 코드 중 
```
cap = cv2.VideoCapture(0)
```
cv2.VideoCapture()을 사용할 수 없다. 왜냐하면 Google Colab은 클라우드에서 코드를 실행하기 때문. 
그래서 이 문제는 해겨하려면 시간이 더 필요할 것 같다.
아래 url 참고.
https://androidkt.com/how-to-capture-and-play-video-in-google-colab/ 