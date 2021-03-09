# multitask-filter
우선 확인하고자 하는 가설은 **"model capacity가 충분하지 않은 경우, multitask 모델이 도달할 수 있는 optimum이 singletask 모델이 도달할 수 있는 optimum보다 나쁘다"** 입니다. (이유는 각 task에 해당하는 loss들을 모두 summation하는 과정에서 conflict가 발생하기 때문)

이를 확인하기 위해 세팅한 실험에 대해서 간단하게만 설명드리자면 아래와 같이 discrete한 인풋에 대해서 무작위로 생성된 2비트 숫자의 아웃풋으로 매핑하는 function을 만들 때
```
1 - 1001010101
2 - 0010101100
3 - 1100110101
...
```
각 자리수에 해당하는 숫자를 각각의 모델이 독립적으로 학습하는 경우(**singletask** 모델; 즉, 위 예시에서는 10개의 독립적인 모델이 각 자리수에 오는 숫자들만 학습하게됨)에 비해
모든 숫자를 한꺼번에 학습하는 모델(**multitask** 모델; 위 예시의 경우에 하나의 모델이 10자리수의 loss를 모두 합한 loss로 학습됨)의 성능이 더 낮게 나옵니다.

3-layer MLP를 기준으로 실험을 진행하였고, 디테일한 하이퍼파라미터는 지난번 공유해드린 코드에서 binary.sh를 참고하시면 될 것 같습니다.
```python
mode='st' # 모델의 모드 설정 (st: singletask, mt: multitask)
p_lossdrop=0.1 # dropout ratio
lr=1e-3 # learning rate
epoch=10000 # number of epochs
batch=512 # batch size
len=10 # 아웃풋에 해당하는 2bit 숫자의 길이
layers=3 # MLP layer의 개수
in_dim=16 # 인풋에 해당하는 숫자의 길이
hid_dim=1024 # MLP의 hidden dimension
seed=0 # random seed (reproducibility를 위해서)
```
위에가 default로 설정한 세팅이고, mode='st'를 입력하면 위에서 이야기한 singletask, mode='mt'를 입력하면 multitask를 수행하게 됩니다.

아래는 위의 세팅에서 간단하게 실험했던 기록입니다.
고정된 hidden dimension으로 st와 mt의 성능을 비교하였고, 총 10000번의 epoch동안 매 epoch마다 loss를 측정해서 최저 loss를 달성한 epoch을 기록하였습니다.
```
hidden_dim=2^14 // st는 9936번째 epoch에서 0.0678, mt는 9733번째 epoch에서 0.0615를 달성 (차이: -0.0063).
hidden_dim=2^13 // st는 9936번째 epoch에서 0.0607, mt는 9733번째 epoch에서 0.0586를 달성 (차이: -0.0021).
hidden_dim=2^12 // st는 9649번째 epoch에서 0.0594, mt는 9394번째 epoch에서 0.0564를 달성 (차이: -0.0030).
hidden_dim=2^11 // st는 9622번째 epoch에서 0.0571, mt는 9859번째 epoch에서 0.0588를 달성 (차이: 0.0017).
hidden_dim=2^10 // st는 9917번째 epoch에서 0.0205, mt는 9961번째 epoch에서 0.0264를 달성 (차이: 0.0059)
hidden_dim=2^7 // st는 9372번째 epoch에서 0.0486, mt는 9992번째 epoch에서 0.0697를 달성 (차이: 0.0211).
hidden_dim=2^5 // st는 9977번째 epoch에서 0.0482, mt는 9842번째 epoch에서 0.0986를 달성 (차이: 0.0504).
hidden_dim=2^4 // st는 9550번째 epoch에서 0.0520, mt는 9621번째 epoch에서 0.1265를 달성 (차이: 0.0745).
hidden_dim=2^3 // st는 9933번째 epoch에서 0.0567, mt는 10000번째 epoch에서 0.1729를 달성 (차이: 0.1162).
```
위 실험에서 hidden dimension의 크기가 작을 때 (즉, 모델 capacity가 충분하지 않을 때) mt(multitask)와 st(singletask) 간의 성능 차이가 점점 더 커지는것을 확인할 수 있습니다.
반면에 hidden dimension의 크기가 매우 커지면 mt와 st의 차이는 점점 줄어들다가 오히려 mt가 역전하기도 합니다. (하지만 이 경우는 고려 대상이 아니므로 배제하면 됩니다.)
