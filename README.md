# facial_expression
😂 딥러닝 기반 사람의 감정 분석

### Data
```fer2013.csv``` 파일 이용  
참고: [FER2013 Dataset](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)

- label = Anger, Disgust, Fear, Happy, Sad, Suprise, Neutral


### Model
- CNN: 정확도
- ResNet_v1 : 정확도 0.3
- inception_resnet_v1
- mtcnn
- VGG19
- ResNet18

### 실행
``` 
python facial_expression.py --cascade haarcascade_frontalface_default.xml --model resnet_model_filter.h5 
```
