import socket
import cv2
import numpy as np

# 서버 설정
HOST = '127.0.0.1'
PORT = 50500

# 소켓 설정 및 연결
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen(10)
print(f"Server listening on {HOST}:{PORT}")

conn, addr = s.accept()
print(f"Connected by {addr}")

capture = cv2.VideoCapture(0)  # 카메라 인덱스를 0으로 설정

if not capture.isOpened():
    print("Camera failed to properly initialize!")
    exit()

while True:
    # 비디오 프레임 캡처
    ret, own_video = capture.read()
    if not ret:
        print("Failed to capture video frame")
        break

    own_video_encode = cv2.imencode('.jpg', own_video)[1].tobytes()
    data_length = len(own_video_encode)

    # 데이터 길이를 먼저 전송
    conn.sendall(data_length.to_bytes(4, 'big'))
    # 실제 이미지 데이터를 전송
    conn.sendall(own_video_encode)

    print(f"Sent frame size: {data_length}")

    # 비디오 데이터 수신
    data_length = int.from_bytes(conn.recv(4), 'big')
    Video_Receive = b""
    while len(Video_Receive) < data_length:
        packet = conn.recv(data_length - len(Video_Receive))
        if not packet:
            break
        Video_Receive += packet

    if len(Video_Receive) == 0:
        print("No data received")
        continue

    print(f"Received data size: {len(Video_Receive)}")

    # 수신 데이터를 배열로 변환
    Video_Receive_in_array = np.frombuffer(Video_Receive, np.uint8)
    
    # 이미지 디코딩
