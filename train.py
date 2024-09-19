import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from uitls import *


# Đường dẫn tới thư mục chứa dữ liệu
DATA_PATH = os.path.join('MP_Data')

# Các hành động sẽ được nhận diện (actions)
actions = load_actions("actions.txt")

# Số lượng video
no_sequences = 30

# Độ dài của mỗi sequence (số frame mỗi video)
sequence_length = 30

# Tạo nhãn cho mỗi hành động
label_map = {label: num for num, label in enumerate(actions)}

# Tạo danh sách sequences và labels
sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            # Đọc file .npy tương ứng
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

# Chuyển đổi dữ liệu thành numpy array
X = np.array(sequences)
y = to_categorical(labels).astype(int)

# Chia dữ liệu thành tập train và test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# Khởi tạo TensorBoard để lưu log
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Xây dựng mô hình LSTM
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

# Compile mô hình
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Huấn luyện mô hình
model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback])

# Lưu mô hình sau khi huấn luyện
model.save('action_recognition_model.h5')

# Đánh giá mô hình trên tập test
loss, accuracy = model.evaluate(X_test, y_test)

print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")
