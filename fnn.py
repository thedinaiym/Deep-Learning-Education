import numpy as np

# activation f-n
def relu(x):
    return np.maximum(0, x)

# derivative for backpropag
def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# for output layer
def sigmoid(x):
    return 1/(1 + np.exp(-x))\

# derivative for backpropag  
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]]
    )
y = np.array([[0], [1], [1], [0]])

# Initialization of weight
np.random.seed(42)
W1 = np.random.randn(2, 2)  # Веса для скрытого слоя (2 входа -> 2 нейрона)
b1 = np.zeros((1, 2))   # Смещение для скрытого слоя
W2 = np.random.randn(2, 1)   # Веса для выходного слоя (2 нейрона -> 1 выход)
b2 = np.zeros((1, 1))

learning_rate = 0.1
epochs = 10000

for epoch in range(epochs):
    # Прямое распространение - Forward Pass
    z1 = np.dot(X, W1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)

    # Среднеквадр-я ошибка
    loss = np.mean((y - a2) ** 2)

    # Обратное распротсранение - Backward Pass
    dz2 = a2 - y  # Производная функции потерь по выходу
    dW2 = np.dot(a1.T, dz2)  # Градиент для W2
    db2 = np.sum(dz2, axis=0, keepdims=True)  # Градиент для b2
    
    dz1 = np.dot(dz2, W2.T) * relu_derivative(z1)  # Производная по скрытому слою
    dW1 = np.dot(X.T, dz1)  # Градиент для W1
    db1 = np.sum(dz1, axis=0, keepdims=True)  # Градиент для b1

    # Обновление весов
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Проверка результата после обучения
print("Predictions:")
print(a2.round())
