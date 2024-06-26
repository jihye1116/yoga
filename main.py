import socket
import cv2
import numpy as np
import threading
import requests  # 예시: HTTP 요청을 보내기 위한 라이브러리

# OpenCV에서 제공하는 얼굴 검출기 XML 파일 경로
CASCADE_PATH = 'haarcascade_frontalface_default.xml'

AI_MODEL_ENDPOINT = 'http://example.com/predict'

# 클라이언트 설정
HOST = '127.0.0.1'
PORT = 50500

# 소켓 설정
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 얼굴 검출기 초기화
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# 서버 역할을 수행하는 함수
def run_server():
    # 소켓 바인딩 및 리스닝
    sock.bind((HOST, PORT))
    sock.listen(10)
    print(f"Server listening on {HOST}:{PORT}")

    while True:
        conn, addr = sock.accept()
        print(f"Connected by {addr}")

        # 클라이언트와 통신할 스레드 생성
        threading.Thread(target=handle_client, args=(conn,)).start()

# 클라이언트 처리 함수
def handle_client(conn):
    while True:
        # 클라이언트로부터 동영상 프레임 수신
        data = conn.recv(921600)
        frame = np.frombuffer(data, dtype=np.uint8).reshape(480, 640, 3)
        
        if len(data) == 0:
            print("No data received")
            continue

        # 얼굴 검출 수행
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # 각 얼굴에 대해 처리
        for (x, y, w, h) in faces:
            # 얼굴 주변에 사각형 그리기
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # AI 모델에 전달할 이미지 데이터 준비
            _, encoded_frame = cv2.imencode('.jpg', frame)
            image_data = {'image': encoded_frame.tobytes()}

            # AI 모델에서 정확도와 코칭 메시지 가져오기
            accuracy, coaching_message = get_ai_results(image_data)
            
            # 클라이언트로 보낼 메시지 준비
            message = f"Accuracy: {accuracy:.2f}, Coaching: {coaching_message}"
            
            # 결과를 클라이언트에 전송
            conn.sendall(message.encode())

        # 처리된 프레임을 클라이언트에 전송
        encoded_frame = cv2.imencode('.jpg', frame)[1].tobytes()
        conn.sendall(encoded_frame)

        # ESC 키를 누르면 종료
        if cv2.waitKey(1) == 27:
            break

    conn.close()

# AI 모델에 이미지를 전송하여 결과를 받아오는 함수
def get_ai_results(image_data):
    # 예시: POST 요청을 사용하여 이미지 데이터를 AI 모델에 전달
    response = requests.post(AI_MODEL_ENDPOINT, data=image_data)
    
    # 예시: JSON 형식의 응답에서 정확도와 코칭 메시지 추출
    result = response.json()
    accuracy = result.get('accuracy')
    coaching_message = result.get('coaching_message')
    
    return accuracy, coaching_message

# 클라이언트 역할을 수행하는 함수
def run_client():
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

# 메인 함수: 서버와 클라이언트 스레드 시작
if __name__ == "__main__":
    server_thread = threading.Thread(target=run_server)
    client_thread = threading.Thread(target=run_client)
    
    server_thread.start()
    client_thread.start()
    
    server_thread.join()
    client_thread.join()
