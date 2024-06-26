import socket
import cv2
import numpy as np

# 클라이언트 설정
HOST = '127.0.0.1'
PORT = 50500

# 소켓 설정 및 연결
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((HOST, PORT))

print(f"Connected to {HOST}:{PORT}")

while True:
    # 서버로부터 동영상 프레임 수신
    data = sock.recv(921600)
    frame = np.frombuffer(data, dtype=np.uint8).reshape(480, 640, 3)

    if len(data) == 0:
        print("No data received")
        continue

    # 서버로부터 정확도와 코칭 메시지 수신
    message = sock.recv(1024).decode()
    print(f"Received message: {message}")

    # 프레임 표시
    cv2.imshow('Video from Server', frame)

    # ESC 키를 누르면 종료
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
sock.close()
