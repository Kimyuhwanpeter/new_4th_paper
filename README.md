# new_4th_paper
* 제안방법 다시 연구해야함
* 본컴에서도 돌려볼것

* 현재 loss들을 모두 수정한상태, object, crop, and weed plain은 유지한 상태임
* Work Home --> 현재 집에서 연구중이였던것
* 아직까지도 진전이 없음, 교수님과 상의 후 다시 진행해야할듯
* **시간은 걸리더라도, 지금 학습하고있는건 조금씩 성능 개선도가 보임 (현재 학습하고있음, MIOU: 74 % 까지 증가)**
* sub_com_train2 이 현재까지 했던 결과물중에 가장 개선됨 (weed iou만 개선하면 됨) --> Generalied dice loss 사용
* Colab_train.py 에서는 batch 를 약간 무리해서 설정했음 --> loss의 alpha 값들은 0.2 or 0.8로 고정되도록 변경했음
<br/>

* Main com 성능은 main_com_train.py 기준으로 잡자
* sub_com_train.py --> around 0.856 
* Currently, Temp_train.py is running on main computer
* 2022년 2월 21일 월요일에 temp_train.py and sub_train.py 확인해볼것 (지금 본컴과 서브컴에서 학습하는중)
