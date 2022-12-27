from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

data = pd.read_csv("dataset.csv")
x=data[1:-1]                
y=data[['target']]              
x_train_all, x_test, y_train_all, y_test = \
  train_test_split(x,y,stratify=y,test_size=0.2,random_state=42)  # 훈련 데이터와 테스트 데이터 분류
x_train, x_val, y_train, y_val = \
  train_test_split(x_train_all,y_train_all,stratify=y_train_all, \
                   test_size=0.2,random_state=42)  # 훈련 데이터와 검증 데이터 분류

scaler = StandardScaler()   # 객체 만들기
scaler.fit(x_train)     # 변환 규칙을 익히기
x_train_scaled = scaler.transform(x_train)  # 데이터를 표준화 전처리
x_val_scaled = scaler.transform(x_val)      # 데이터를 표준화 전처리

mlp = MLPClassifier(hidden_layer_sizes=(10,), activation='logistic', \
                    solver='sgd', alpha=0.01, batch_size=32, \
                    learning_rate_init=0.1, max_iter=500)  # 객체 생성

mlp.fit(x_train_scaled, y_train)    # 훈련하기
mlp.score(x_val_scaled, y_val)      # 정확도 평가