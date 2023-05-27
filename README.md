# Analysis the learning environment of CNN-based image classification model
- task : Image Classification
- dataset : CIFAR10
- model : AlexNet, VGGNet, ResNet

## abstract
Convolutional Neural Network (CNN)의 등장을 기점으로 deep learning은 다양한 분야에서 급속도로 발전하게 된다. CNN 모델에서 우리가 직접 정해야하는 변수를 hyper parameter라고 하는데, 높은 정확도를 도출하기 위해서는 모델의 아키텍쳐뿐만 아니라 모델의 학습환경(hyper parameter 설정값)도 중요하다. 본 연구에서는 CNN 기반 image classification 모델 중 AlexNet, VGGNet-16, ResNet-50, ResNet-101로 CIFAR-10 dataset을 이용하여 실험한다. Data augmentation(1), batch size(2), learning rate(3)으로 총 3가지의 실험을 진행하고 정확도의 변화를 분석한다.

## experiment
1. CIFAR10 에 맞게 모델 아키텍쳐 수정
2. data augmentation 
3. batch size
4. learning rate

## More details
[AJOU SOFTCON 2022-2](https://softcon.ajou.ac.kr/works/works.asp?uid=711&category=M) <br/><br/>
<img width="720" alt="softcon_허혜진" src="https://user-images.githubusercontent.com/87194339/232229677-ebfc03fa-21ff-4e84-a7b6-940e919a2f57.png">
