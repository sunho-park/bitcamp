#1. 데이터
import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15])
y_test = np.array([11,12,13,14,15])
x_pred = np.array([16,17,18])

# 2. 모델구성
from keras.models import Sequential
from keras.layers import Dense

# Sequential 함수를 model 로 하겠다.

model = Sequential()
  

model.add(Dense(5, input_dim=1))
model.add(Dense(3))
# model.add(Dense(1000000))
# model.add(Dense(1000000))
# model.add(Dense(1000000))
# model.add(Dense(1000000))
# model.add(Dense(1000000))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(500))
model.add(Dense(1))

# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=50, batch_size=1) #훈련용 데이터로 훈련

# 4. 평가, 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1) #훈련용 데이터와 평가용 데이터를 분리해야함
print("loss : ", loss)
print("mse = ", mse)

y_pred = model.predict(x_pred)
print("y_predict : \n", y_pred)
