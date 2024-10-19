import torch
import torch.nn as nn
import torch.optim as optim

# X: входные последовательности, y: целевые значения (сдвиг на 1 позицию)
X = torch.tensor([[0.0, 1.0, 2.0],
                  [1.0, 2.0, 3.0],
                  [2.0, 3.0, 4.0]], dtype=torch.float32).unsqueeze(-1)  # (batch_size, seq_len, input_size)

y = torch.tensor([3.0, 4.0, 5.0], dtype=torch.float32)

# Определение архитектуры RNN
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)  # RNN слой
        self.fc = nn.Linear(hidden_size, output_size)  # Полносвязанный слой

    def forward(self, x):
        out, _ = self.rnn(x)  # Прямое распространение через RNN слой
        out = self.fc(out[:, -1, :])  # Используем только последний выход из RNN
        return out

# Инициализация модели, функции потерь и оптимизатора
input_size = 1
hidden_size = 10
output_size = 1
rnn = RNN(input_size, hidden_size, output_size)
criterion = nn.MSELoss()  # Среднеквадратичная ошибка для регрессии
optimizer = optim.Adam(rnn.parameters(), lr=0.01)

# Обучение
epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()  # Обнуляем градиенты
    outputs = rnn(X)  # Прямое распространение
    loss = criterion(outputs, y.unsqueeze(-1))  # Вычисление функции потерь
    loss.backward()  # Обратное распространение
    optimizer.step()  # Обновление весов
    
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

print('Finished Training')
