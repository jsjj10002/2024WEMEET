import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from deepface import DeepFace
import re
import csv
import json
from glob import glob
import shutil
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from google.cloud import vision
import ast
import openai
import sys
from datetime import datetime
from tqdm import tqdm
# 경고 메시지 관련 라이브러리
import warnings
import logging

#미디어파이프 초기화
mp_hands = mp.solutions.hands

# ResNet 모델 경로를 전역변수로 지정
global MODEL_PATH
MODEL_PATH = ""
# 구글 클라우드 인증 키 경로
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""  
#openai api key
client = openai.OpenAI(api_key="")

# 경고 메시지 숨기기


# 경고 메시지 숨기기 설정
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # TensorFlow 경고 숨기기
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # TensorFlow 로그 레벨 설정
logging.getLogger('mediapipe').setLevel(logging.ERROR)  # MediaPipe 로그 레벨 설정


#video_path = 'test.mp4'  # 동영상 파일 경로
#output_dir = '/mp4_to_image'

# 감정: vidieo_path -> /mp4_to_image -> /image_crop -> /predict -> /predict/hamburger -> /predict_google_api -> /final_output -> /deepface_emotions
# 행동: vidieo_path , /final_output -> "movement_analysis_results.csv"


# 1. 비디오 불러와서 프레임 단위로 이미지 
def extract_frames(video_path, output_dir="mp4_to_image"):
    """
    비디오에서 1초 간격으로 프레임을 추출하여 이미지로 저장하는 함수
    Args:
        video_path (str): 입력 비디오 파일 경로
        output_dir (str): 프레임 이미지를 저장할 디렉토리 경로
    """
    # 저장할 디렉토리 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 비디오 파일 읽기
    video = cv2.VideoCapture(video_path)

    # 비디오 정보 가져오기
    fps = video.get(cv2.CAP_PROP_FPS)  # 초당 프레임 수 (frame per second)
    count = 0

    while video.isOpened():
        ret, frame = video.read()
        
        if not ret:
            break

        # 현재 프레임 번호 가져오기
        frame_id = int(video.get(cv2.CAP_PROP_POS_FRAMES))

        # 1초마다 프레임을 저장 (fps에 따라 결정)
        if frame_id % int(fps) == 0:
            filename = f"{output_dir}/picture_{count}.jpg"
            cv2.imwrite(filename, frame)
            #print(f"Saved {filename}")
            count += 1
    # 비디오 객체 해제
    video.release()

# 2. 손 기준으로 이미지 크롭
class HandProcessor:
    """
    손 감지 및 이미지 크롭을 처리하는 클래스
    """
    def __init__(self):
        """
        HandProcessor 클래스 초기화
        """
        self.mp_hands = mp.solutions.hands
        
    def calculate_average_point(self, landmarks, image_width, image_height):
        """
        손 랜드마크의 평균 좌표를 계산합니다.
        """
        x_coords = [lm.x * image_width for lm in landmarks]
        y_coords = [lm.y * image_height for lm in landmarks]
        
        avg_x = int(np.mean(x_coords))
        avg_y = int(np.mean(y_coords))
        
        return avg_x, avg_y

    def crop_and_save_image(self, frame, center_x, center_y, original_filename, hand_label, output_dir = "image_crop", size=500):
        """
        이미지를 중심 좌표로 크롭하고 저장합니다.
        """
        half_size = size // 2
        height, width = frame.shape[:2]
        
        # 크롭 영역 계산
        x1 = max(0, center_x - half_size)
        y1 = max(0, center_y - half_size)
        x2 = min(width, center_x + half_size)
        y2 = min(height, center_y + half_size)
        
        if x1 >= x2 or y1 >= y2:
            return
            
        # 이미지 크롭
        cropped = frame[y1:y2, x1:x2]
        
        # 원본 파일명에서 확장자 제거
        base_filename = os.path.splitext(original_filename)[0]
        # 새로운 파일명 형식으로 저장 (원본파일명_left 또는 _right.jpg)
        filename = f"{base_filename}_{hand_label}.jpg"
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, cropped)

    def process(self, input_dir = "mp4_to_image", output_dir = "image_crop"):
        """
        입력 디렉토리의 이미지들에서 손을 감지하고 크롭하여 출력 디렉토리에 저장합니다.
        
        Args:
            input_dir (str): 입력 이미지가 있는 디렉토리 경로
            output_dir (str): 크롭된 이미지를 저장할 디렉토리 경로
        """
        os.makedirs(output_dir, exist_ok=True)
        with self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2, 
            min_detection_confidence=0.5
        ) as hands:
            # 입력 폴더에서 모든 이미지 파일 가져오기
            image_files = glob(os.path.join(input_dir, '*.jpg')) + \
                         glob(os.path.join(input_dir, '*.png'))
            
            for image_path in image_files:
                # 원본 파일명 추출
                original_filename = os.path.basename(image_path)
                
                # 이미지 읽기
                frame = cv2.imread(image_path)
                if frame is None:
                    #print(f"이미지를 읽을 수 없습니다: {image_path}")
                    continue
                    
                # RGB 변환 및 손 좌표 탐지
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)
                
                if results.multi_hand_landmarks and results.multi_handedness:
                    image_height, image_width = frame.shape[:2]
                    
                    # 각 손에 대해 처리
                    for hand_idx, (hand_landmarks, handedness) in enumerate(
                        zip(results.multi_hand_landmarks, results.multi_handedness)
                    ):
                        # 중심점 계산 (손가락 관절 좌표의 평균)
                        center_x, center_y = self.calculate_average_point(
                            hand_landmarks.landmark, image_width, image_height
                        )
                        
                        # 왼손 또는 오른손 라벨 가져오기
                        hand_label = "left" if handedness.classification[0].label == "Left" else "right"
                        
                        # 이미지 크롭 및 저장 
                        self.crop_and_save_image(frame, center_x, center_y, original_filename, hand_label, output_dir = "image_crop")


# 3. resnet 모델 이용한 예측
class HamburgerClassifier:
    """
    ResNet 모델을 사용하여 이미지가 햄버거인지 아닌지 분류하는 클래스
    """
    def __init__(self):
        # 클래스 초기화
        self.classes = ['hamburger', 'not hamburger']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 32
        
        # 모델 초기화
        self.model = models.resnet18()
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, len(self.classes))
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 전처리 파이프라인 설정
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def process_batch(self, all_images):
        """배치 단위로 이미지 처리"""
        input_batch = torch.cat(all_images).to(self.device)
        with torch.no_grad():
            outputs = self.model(input_batch)
            _, predicted = torch.max(outputs, 1)
        return predicted

    def classify_and_copy(self, all_images, all_files, input_dir= "image_crop", output_dir= "predict"):
        """이미지 분류 및 결과 폴더로 복사"""
        if all_images:
            predicted = self.process_batch(all_images)
            for i, file in enumerate(all_files):
                label = self.classes[predicted[i].item()]
                destination_folder = os.path.join(output_dir, 'hamburger' if label == 'hamburger' else 'not_hamburger')
                os.makedirs(destination_folder, exist_ok=True)
                destination = os.path.join(destination_folder, file)
                
                try:
                    shutil.copy(os.path.join(input_dir, file), destination)
                    #print(f"이미지: {file}, 예측 결과: {label}, 저장 위치: {destination}")
                except FileNotFoundError:
                    #print(f"파일이 존재하지 않습니다: {file}")
                    pass

    def process(self, input_dir= "image_crop", output_dir= "predict"):
        """
        입력 디렉토리의 이미지들을 처리하여 햄버거/비햄버거로 분류
        
        Args:
            input_dir (str): 입력 이미지가 있는 디렉토리 경로
            output_dir (str): 분류된 이미지를 저장할 디렉토리 경로
        """
        os.makedirs(output_dir, exist_ok=True)
        
        all_images, all_files = [], []
        for image_file in os.listdir(input_dir):
            if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            image_path = os.path.join(input_dir, image_file)
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0)
            all_images.append(input_tensor)
            all_files.append(image_file)

            if len(all_images) >= self.batch_size:
                self.classify_and_copy(all_images, all_files, input_dir= "image_crop", output_dir= "predict")
                all_images, all_files = [], []

        # 남은 이미지 처리
        if all_images:
            self.classify_and_copy(all_images, all_files, input_dir= "image_crop", output_dir= "predict")
            
            
# 4. google vision api 이용
class EatingActivityDetector:
    """
    Google Vision API를 사용하여 이미지에서 식사 행동을 감지하는 클래스
    """
    def __init__(self):
        """
        EatingActivityDetector 클래스 초기화
        """
        self.client = vision.ImageAnnotatorClient()

    def detect_eating_activity(self, image_path):
        """
        이미지에서 식사 행동 감지
        
        Args:
            image_path (str): 분석할 이미지 파일 경로
            
        Returns:
            dict: eating_score와 food_score를 포함하는 딕셔너리
        """
        with open(image_path, 'rb') as image_file:
            content = image_file.read()
        
        image = vision.Image(content=content)
        features = [vision.Feature.Type.LABEL_DETECTION]
        
        request = vision.AnnotateImageRequest(
            image=image,
            features=[vision.Feature(type_=feature) for feature in features]
        )
        
        response = self.client.annotate_image(request=request)
        
        # Eating과 Food에 대해서만 확인
        eating_score = 0.0
        food_score = 0.0
        
        for label in response.label_annotations:
            if label.description.lower() == 'eating':
                eating_score = label.score
            elif label.description.lower() == 'food':
                food_score = label.score
        
        return {
            'eating_score': eating_score,
            'food_score': food_score
        }

    def process(self, input_dir= "predict/hamburger", output_dir= "predict_google_api", confidence_threshold=0.75):
        """
        입력 폴더의 이미지들에 대해 식사 행동을 감지하고 임계값을 넘는 이미지를 출력 폴더로 복사
        
        Args:
            input_dir (str): 입력 이미지가 있는 디렉토리 경로
            output_dir (str): 선별된 이미지를 저장할 디렉토리 경로
            confidence_threshold (float): 식사 행동 감지 임계값 (기본값: 0.75)
        """
        # 출력 폴더가 없으면 생성
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 지원하는 이미지 확장자
        valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
        
        try:
            # 입력 폴더의 모든 파일 처리
            for filename in os.listdir(input_dir):
                if any(filename.lower().endswith(ext) for ext in valid_extensions):
                    input_path = os.path.join(input_dir, filename)
                    
                    try:
                        # Google Vision API로 식사 행동 감지
                        result = self.detect_eating_activity(input_path)
                        
                        # Eating 점수가 임계값보다 높으면 이미지 복사
                        if result['eating_score'] >= confidence_threshold:
                            output_path = os.path.join(output_dir, filename)
                            shutil.copy2(input_path, output_path)
                            #print(f"복사됨: {filename} (Eating 점수: {result['eating_score']:.2%})")
                        else:
                            #print(f"건너뜀: {filename} (Eating 점수: {result['eating_score']:.2%})")
                            pass
                            
                    except Exception as e:
                        #print(f"이미지 처리 중 오류 발생 ({filename}): {str(e)}")
                        pass

            #print("\n처리가 완료되었습니다.")
                
        except Exception as e:
            #print(f"전체 처리 중 오류 발생: {str(e)}")
            pass


# 5. mediapipe 이용한 얼굴
class FaceHandProcessor:
    """
    얼굴과 손의 위치를 감지하여 거리를 계산하고 이미지를 처리하는 클래스
    """
    def __init__(self):
        """
        FaceHandProcessor 클래스 초기화
        """
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh
        self.hands = self.mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
        self.distance_threshold = 450

    def get_base_name(self, filename):
        """
        _left, _right 접미어가 있으면 제거하고 기본 파일명만 반환
        """
        filename = filename.rsplit('.', 1)[0]  # 확장자 제거
        if filename.endswith('_left'):
            return filename[:-5]  # '_left' 제거
        elif filename.endswith('_right'):
            return filename[:-6]  # '_right' 제거
        return filename

    def process(self, input_dir= "predict_google_api", output_dir= "final_output", source_dir= "mp4_to_image", ori_dir= "find_original_image"):
        """
        이미지 처리 메인 프로세스
        Args:
            input_dir (str): 입력 이미지 디렉토리 경로
            output_dir (str): 최종 출력 이미지 디렉토리 경로  
            source_dir (str): 원본 이미지 디렉토리 경로
            ori_dir (str): 중간 처리 이미지 디렉토리 경로
        """
        # 필요한 폴더 생성
        for folder in [ori_dir, output_dir]:
            if not os.path.exists(folder):
                os.makedirs(folder)

        # Step 1: input 폴더의 기본 이름 목록 생성
        input_files = [f for f in os.listdir(input_dir) if f.endswith('.jpg')]
        input_basenames = list(set(self.get_base_name(f) for f in input_files))
        print("Updated input_basenames:", input_basenames)

        # Step 2: source 폴더에서 매칭되는 이미지 찾아서 ori_dir로 복사
        source_files = [f for f in os.listdir(source_dir) if f.endswith('.jpg')]
        found_count = 0

        for source_file in source_files:
            source_base = self.get_base_name(source_file)

            if source_base in input_basenames:
                source_path = os.path.join(source_dir, source_file)
                dest_path = os.path.join(ori_dir, source_file)
                try:
                    shutil.copy2(source_path, dest_path)
                    found_count += 1
                    #print(f"Copied {source_file} to {ori_dir}")
                except Exception as e:
                    #print(f"Error copying {source_file}: {e}")
                    pass
            else:
                #print(f"Skipping {source_file}, not found in input_basenames")
                pass

            #print(f"Found and copied {found_count} matching images to {ori_dir}")

        # Step 3: ori_dir의 이미지들에 대해 거리 계산 및 조건에 맞는 이미지 복사
        found_files = [f for f in os.listdir(ori_dir) if f.endswith('.jpg')]
        processed_count = 0
        saved_count = 0

        for img_file in found_files:
            img_path = os.path.join(ori_dir, img_file)
            frame = cv2.imread(img_path)
            
            if frame is None:
                #print(f"Failed to read image: {img_file}")
                continue

            # 좌표 추출
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 코 좌표 추출
            face_results = self.face_mesh.process(frame_rgb)
            nose_coordinates = None
            if face_results.multi_face_landmarks:
                landmarks = face_results.multi_face_landmarks[0].landmark
                height, width, _ = frame.shape
                nose_coordinates = (int(landmarks[4].x * width), int(landmarks[4].y * height))

            # 손목 좌표 추출
            hand_results = self.hands.process(frame_rgb)
            wrist_coordinates = None
            if hand_results.multi_hand_landmarks:
                landmarks = hand_results.multi_hand_landmarks[0].landmark
                height, width, _ = frame.shape
                wrist_coordinates = (int(landmarks[0].x * width), int(landmarks[0].y * height))

            # 거리 계산 및 조건 확인
            if wrist_coordinates is not None and nose_coordinates is not None:
                distance = np.sqrt((nose_coordinates[0] - wrist_coordinates[0]) ** 2 + 
                                (nose_coordinates[1] - wrist_coordinates[1]) ** 2)

                if distance <= self.distance_threshold:
                    saved_count += 1
                    output_path = os.path.join(output_dir, img_file)
                    try:
                        shutil.copy2(img_path, output_path)
                        #print(f"Image {img_file}: Distance = {distance:.2f} (Saved)")
                    except Exception as e:
                        #print(f"Error copying {img_file}: {e}")
                        pass
            
            processed_count += 1
            
# 6. 행동 분석: positive, negative, negative behavior 리턴함.
class MovementAnalyzer:
    def __init__(self):
        """
        MovementAnalyzer 클래스 초기화
        MediaPipe Pose 모델을 초기화합니다.
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()

    def classify_movement_probabilities(self, cumulative_x, cumulative_y, th_1=2.0, th_2=0.5, noise_threshold_x=0.01, noise_threshold_y=0.01):
        """
        x 변화량 / y 변화량 비율을 기준으로 움직임의 확률을 계산합니다.
        """
        movements = {
            "positive": 0,
            "negative": 0,
            "neutral": 0
        }
        total_frames = len(cumulative_x)

        for i in range(1, total_frames):
            delta_x = abs(cumulative_x[i] - cumulative_x[i - 1])
            delta_y = abs(cumulative_y[i] - cumulative_y[i - 1])

            # 노이즈 제거
            if delta_x < noise_threshold_x and delta_y < noise_threshold_y:
                movements["neutral"] += 1
            else:
                a = delta_x / (delta_y + 1e-6)
                if a > th_1:
                    movements["negative"] += 1
                elif a < th_2:
                    movements["positive"] += 1
                else:
                    movements["neutral"] += 1

        # 총 움직임 수
        total_movement = movements["positive"] + movements["negative"]
        
        if total_movement > 0:
            probabilities = {
                "positive": (movements["positive"] / total_movement) * 100,
                "negative": (movements["negative"] / total_movement) * 100,
                "neutral": 0  # 중립은 제외
            }
        else:
            probabilities = {"positive": 50, "negative": 50, "neutral": 0}  # 기본값 설정
        
        return probabilities

    def process_video_with_probabilities(self, video_path, start_time, duration=10, fps=30, th_1=2.0, th_2=0.5, noise_threshold_x=0.01, noise_threshold_y=0.01):
        """
        입력 영상에서 특정 시간 구간의 움직임을 확률 기반으로 분석합니다.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            #print(f"Error: Could not open video file {video_path}")
            return None, None, None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_frame = int(start_time * fps)
        end_frame = min(start_frame + int(duration * fps), total_frames)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        nose_positions = []
        cumulative_x = [0]
        cumulative_y = [0]
        frame_count = start_frame

        while cap.isOpened() and frame_count < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image)

            if results.pose_landmarks:
                nose = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
                nose_positions.append((nose.x, nose.y))

                if len(nose_positions) > 1:
                    prev_x, prev_y = nose_positions[-2]
                    curr_x, curr_y = nose_positions[-1]

                    delta_x = abs(curr_x - prev_x)
                    delta_y = abs(curr_y - prev_y)

                    cumulative_x.append(cumulative_x[-1] + delta_x)
                    cumulative_y.append(cumulative_y[-1] + delta_y)

            frame_count += 1

        cap.release()

        if len(cumulative_x) <= 1 or len(cumulative_y) <= 1:
            #print(f"No valid movement data detected in video {video_path}")
            return None, None, None

        probabilities = self.classify_movement_probabilities(
            cumulative_x, cumulative_y, th_1, th_2, noise_threshold_x, noise_threshold_y
        )
        return probabilities, cumulative_x, cumulative_y

    def process(self, video_path, input_dir = "final_output", duration=10, fps=30):
        """
        이미지 파일 이름에서 시간을 추출하여 비디오를 분석하고 CSV 파일로 저장합니다.
        """
        image_files = [f for f in os.listdir(input_dir) if f.startswith("picture_") and f.endswith((".jpg", ".png"))]

        results = []  # CSV 저장을 위한 리스트
        positive_values = []
        negative_values = []

        for image_file in image_files:
            try:
                time_str = image_file.split("_")[1].split(".")[0]
                start_time = int(time_str)
            except (IndexError, ValueError):
                #print(f"Invalid file format: {image_file}")
                continue

            #print(f"\nAnalyzing movement for {image_file} (Start Time: {start_time}s)")

            probabilities, cumulative_x, cumulative_y = self.process_video_with_probabilities(
                video_path, start_time=start_time, duration=duration, fps=fps
            )

            if probabilities:
                positive_raw = probabilities['positive']
                negative_raw = probabilities['negative']
                total = positive_raw + negative_raw

                if total > 0:
                    positive_normalized = (positive_raw / total) * 100
                    negative_normalized = (negative_raw / total) * 100
                else:
                    positive_normalized = 50
                    negative_normalized = 50

                positive_values.append(positive_normalized)
                negative_values.append(negative_normalized)

                print(f"Normalized Movement Probabilities (from {start_time}s to {start_time + duration}s):")
                print(f"- Positive (긍정): {positive_normalized:.2f}%")
                print(f"- Negative (부정): {negative_normalized:.2f}%")

                # CSV 저장을 위한 데이터 추가
                results.append({
                    "Image File": image_file,
                    "Start Time (s)": start_time,
                    "Positive (%)": positive_normalized,
                    "Negative (%)": negative_normalized
                })

        positive_behave = sum(positive_values) / len(positive_values) if positive_values else 50
        negative_behave = sum(negative_values) / len(negative_values) if negative_values else 50

        #print("\nFinal Results:")
        #print(f"Average Positive (긍정 평균): {positive_behave:.2f}%")
        #print(f"Average Negative (부정 평균): {negative_behave:.2f}%")

        # CSV 파일 저장
        df = pd.DataFrame(results)
        csv_output_path = "movement_analysis_results.csv"
        df.to_csv(csv_output_path, index=False, encoding="utf-8-sig")

        return positive_behave, negative_behave
        
# 7. deepface 이용
class EmotionAnalyzer:
    """
    DeepFace를 이용한 감정 분석을 수행하는 클래스
    """
    def __init__(self):
        # DeepFace가 반환하는 모든 감정 목록
        self.ALL_EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
    def extract_frame_number(self, filename):
        """
        파일명에서 프레임 번호를 추출하는 함수
        """
        match = re.search(r'_(\d+)\.jpg', filename)
        return int(match.group(1)) if match else None

    def process(self, ori_dir = "mp4_to_image", output_dir = "deepface_emotions", input_dir = "final_output"):
        """
        감정 분석을 수행하고 결과를 저장하는 메인 프로세스
        Args:
            ori_dir (str): 원본 이미지가 있는 디렉토리 경로
            output_dir (str): 결과 파일을 저장할 디렉토리 경로
            input_dir (str): 섭취 순간 이미지가 있는 디렉토리 경로
        """
        # 결과 저장 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 결과 파일 경로 설정
        result_csv = os.path.join(output_dir, "emotion_analysis_full.csv")
        result_json = os.path.join(output_dir, "emotion_analysis_full.json")

        # 섭취 순간 프레임 읽기 및 정렬
        eating_frames = []
        for img_file in os.listdir(input_dir):
            frame_number = self.extract_frame_number(img_file)
            if frame_number is not None:
                eating_frames.append(frame_number)

        eating_frames.sort()

        # 연속된 프레임 묶기
        grouped_frames = []
        current_group = [eating_frames[0]]

        for i in range(1, len(eating_frames)):
            if eating_frames[i] - eating_frames[i - 1] <= 3:  # 간격 3 이하
                current_group.append(eating_frames[i])
            else:
                grouped_frames.append(current_group)
                current_group = [eating_frames[i]]

        grouped_frames.append(current_group)  # 마지막 그룹 추가

        # 감정 분석 결과 저장용 리스트
        analysis_results = []

        # 각 그룹의 전후 프레임 포함하여 DeepFace 감정 분석 수행
        for group in grouped_frames:
            group_result = {
                "group": group,
                "frames_analyzed": [],
                "emotions": {}
            }
            start_frame = max(1, group[0] - 5)  # 전 5프레임
            end_frame = group[-1] + 5           # 후 5프레임
            frames_to_analyze = range(start_frame, end_frame + 1)

            for frame in frames_to_analyze:
                img_file = f"picture_{frame}.jpg"
                img_path = os.path.join(ori_dir, img_file)

                if not os.path.exists(img_path):
                    #print(f"Image {img_file} not found, skipping.")
                    continue

                try:
                    # DeepFace 분석 수행
                    analysis = DeepFace.analyze(img_path, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
                    if isinstance(analysis, list):
                        analysis = analysis[0]

                    # 결과 저장
                    group_result["frames_analyzed"].append(frame)
                    group_result["emotions"][frame] = {
                        "dominant_emotion": analysis.get("dominant_emotion", "Unknown"),
                        "emotion_probabilities": {emotion: analysis.get("emotion", {}).get(emotion, 0) for emotion in self.ALL_EMOTIONS}
                    }

                    #print(f"Frame {frame} ({img_file}): {group_result['emotions'][frame]}")

                except Exception as e:
                    #print(f"Error processing {img_file}: {str(e)}")
                    pass

            analysis_results.append(group_result)

        # 주요 감정 통계 계산 및 전후 비교
        for result in analysis_results:
            group = result["group"]
            emotions = result["emotions"]
            emotion_values = list(emotions.values())

            # 감정 확률 평균 계산
            all_emotions = {emotion: [] for emotion in self.ALL_EMOTIONS}
            for frame_data in emotion_values:
                for emotion, prob in frame_data["emotion_probabilities"].items():
                    all_emotions[emotion].append(prob)

            avg_emotions = {emotion: np.mean(probs) if probs else 0 for emotion, probs in all_emotions.items()}
            #print(f"Group {group}: Average Emotions: {avg_emotions}")

            # 긍정/부정 감정 비율 계산
            positive_ratio = avg_emotions.get("happy", 0) + avg_emotions.get("neutral", 0)
            negative_ratio = sum(avg_emotions.get(key, 0) for key in ["fear", "angry", "sad"])
            #print(f"Group {group}: Positive Ratio: {positive_ratio}, Negative Ratio: {negative_ratio}")

        # 결과 CSV로 저장
        with open(result_csv, mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Group", "Frame", "Dominant Emotion", "Emotion Probabilities"])
            for result in analysis_results:
                for frame, data in result["emotions"].items():
                    writer.writerow([
                        result["group"],
                        frame,
                        data["dominant_emotion"],
                        data["emotion_probabilities"]
                    ])

        # 결과 JSON으로 저장
        with open(result_json, mode='w', encoding='utf-8') as jsonfile:
            json.dump(analysis_results, jsonfile, ensure_ascii=False, indent=4)

# 8. 감정 변화 분석
def analyze_emotion_changes():
    """
    감정 분석 결과를 비교하고 전/후 변화를 분석하는 함수
    """
    # CSV 파일 읽기
    csv_file_path = "./deepface_emotions/emotion_analysis_full.csv"  # 업로드된 CSV 경로
    data = pd.read_csv(csv_file_path)

    # 그룹별 섭취 전후 프레임 나누기
    grouped = data.groupby("Group")

    # 결과 저장용 리스트
    comparison_results = []

    for group, group_data in grouped:
        group_data = group_data.sort_values("Frame")  # 프레임 순서대로 정렬
        
        # 최소 7프레임: 전 3프레임, 섭취, 후 3프레임
        if len(group_data) >= 7:
            # 전 3프레임, 섭취, 후 3프레임 나누기
            before_frames = group_data.iloc[:5]
            after_frames = group_data.iloc[-5:]
            
            # 감정 확률 평균 계산
            before_avg = before_frames["Emotion Probabilities"].apply(eval).apply(pd.Series).mean()
            after_avg = after_frames["Emotion Probabilities"].apply(eval).apply(pd.Series).mean()
            
            # 감정 변화 계산
            emotion_changes = after_avg - before_avg
            
            # 저장
            comparison_results.append({
                "Group": group,
                "Before Average": before_avg.to_dict(),
                "After Average": after_avg.to_dict(),
                "Emotion Changes": emotion_changes.to_dict()
            })

            # 출력
            #print(f"Group {group}:")
            #print(f"  Before Average: {before_avg}")
            #print(f"  After Average: {after_avg}")
            #print(f"  Emotion Changes: {emotion_changes}")

    # 결과를 DataFrame으로 변환
    comparison_df = pd.DataFrame(comparison_results)

    # 결과 저장
    comparison_df.to_csv("./deepface_emotions/1emotion_analysis_full.csv", index=False)
    #print("Emotion comparison results saved to 'emotion_comparison_results.csv'.")

# 9. 감정 점수 계산: positive, negative, neutral 점수 리턴함.
def analyze_emotion_scores():
    """
    감정 분석 결과를 계산하여 긍정/부정/중립 점수를 반환하는 함수
    
    Returns:
        tuple: (positive_emotion, negative_emotion, neutral_emotion) 점수
    """
    # CSV 파일 읽기
    csv_file_path = "./deepface_emotions/1emotion_analysis_full.csv"
    data = pd.read_csv(csv_file_path)

    # 비교 결과 리스트 생성
    comparison_results = []

    # CSV에서 데이터를 변환하여 사용
    for _, row in data.iterrows():
        comparison_results.append({
            "Emotion Changes": ast.literal_eval(row["Emotion Changes"])  # Emotion Changes를 딕셔너리로 변환
        })

    # 감정 레이블 정의
    positive_emotions = ["happy"]
    negative_emotions = ["angry", "fear", "sad", "disgust"]
    neutral_emotions = ["neutral"]

    # 총 점수 계산용 변수
    total_positive_score = 0
    total_negative_score = 0
    total_neutral_score = 0

    # 감정 변화 합산
    for result in comparison_results:
        emotion_changes = pd.Series(result["Emotion Changes"])

        # 감정별 점수 합산
        total_positive_score += emotion_changes[positive_emotions].sum()
        total_negative_score += emotion_changes[negative_emotions].sum()
        total_neutral_score += emotion_changes[neutral_emotions].sum()

    # Min–Max 스케일링 수행
    scores = np.array([total_positive_score, total_negative_score, total_neutral_score])
    min_val, max_val = scores.min(), scores.max()

    # 예외처리: max == min일 경우 (스코어 차이가 없을 때)
    if max_val - min_val == 0:
        scaled_scores = np.ones_like(scores) / len(scores)  # 균등하게 33.3%씩 분배
    else:
        scaled_scores = (scores - min_val) / (max_val - min_val)  # Min–Max 스케일링

    # 정규화 (합을 100%로 변환)
    normalized_scores = (scaled_scores / scaled_scores.sum()) * 100

    # 최종 점수 계산
    positive_emotion = normalized_scores[0]
    negative_emotion = normalized_scores[1]
    neutral_emotion = normalized_scores[2]

    return positive_emotion, negative_emotion, neutral_emotion

# 10. 최종 점수 저장 : 변수 선번 부터 해줘야함
def calculate_final_scores(positive_emotion, negative_emotion, neutral_emotion, positive_behave, negative_behave):
    """
    감정과 행동 분석 결과를 종합하여 최종 점수를 계산하는 함수
    Returns:
        dict: Y_positive, Y_negative, Y_neutral 값을 포함하는 딕셔너리
    """
    # 가중치 설정
    a = 0.8  # 감정 분석 가중치
    b = 0.2  # 행동 분석 가중치

    # Y 값 계산
    Y_positive = a * positive_emotion + b * positive_behave
    Y_negative = a * negative_emotion + b * negative_behave 
    Y_neutral = a * neutral_emotion  # 행동에는 중립 값이 없음

    # 결과 데이터 구성
    data = {
        'id': ['Y_positive', 'Y_negative', 'Y_neutral'],
        'value': [Y_positive, Y_negative, Y_neutral]
    }

    # DataFrame 생성 및 CSV 저장
    result_df = pd.DataFrame(data)
    result_df.to_csv("./deepface_emotions/result.csv", index=False)

    return data

# 11. openai 이용 보고서

def analyze_and_save_report(output_json_path):
    """
    감정과 행동 분석 결과를 종합하여 보고서를 생성하고 JSON 파일로 저장하는 함수
    
    Args:
        output_json_path (str): 결과 JSON 파일 저장 경로
    """
    # ✅ 감정 분석 CSV 파일 로드
    emotion_file_path = "./deepface_emotions/1emotion_analysis_full.csv"
    df_emotion = pd.read_csv(emotion_file_path)

    # ✅ 행동 분석 CSV 파일 로드
    behavior_file_path = "movement_analysis_results.csv"
    df_behavior = pd.read_csv(behavior_file_path)

    # ✅ 감정 분석 컬럼 정리
    df_emotion.columns = df_emotion.columns.str.strip()
    for col in ["Before Average", "After Average", "Emotion Changes"]:
        df_emotion[col] = df_emotion[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # ✅ 감정 유형 정의
    positive_emotions = ["happy", "surprise"]
    negative_emotions = ["angry", "disgust", "fear", "sad"]
    neutral_emotions = ["neutral"]

    def categorize_emotions(changes):
        """ 감정을 긍정, 부정, 중립으로 정리하여 100% 정규화 """
        categorized = {"positive": 0, "negative": 0, "neutral": 0}
        for emotion, value in changes.items():
            if emotion in positive_emotions:
                categorized["positive"] += abs(value)
            elif emotion in negative_emotions:
                categorized["negative"] += abs(value)
            elif emotion in neutral_emotions:
                categorized["neutral"] += abs(value)

        total = sum(categorized.values())
        return {k: round((v / total) * 100, 2) if total > 0 else 0 for k, v in categorized.items()}

    # ✅ 그룹 정렬
    df_emotion["Group"] = df_emotion["Group"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df_emotion["Group"] = df_emotion["Group"].apply(lambda x: x[0] if isinstance(x, list) else x)
    df_emotion = df_emotion.sort_values(by="Group")

    # ✅ 그룹별 감정 변화 저장
    group_emotion = []
    total_scores = {"positive": 0, "negative": 0, "neutral": 0}
    emotion_trends = []

    for index, (group, row) in enumerate(df_emotion.iterrows(), start=1):
        changes = row["Emotion Changes"]
        categorized_changes = categorize_emotions(changes)

        trend = "유지됨"
        if sum(row["After Average"].values()) > sum(row["Before Average"].values()):
            trend = "긍정 증가"
        elif sum(row["After Average"].values()) < sum(row["Before Average"].values()):
            trend = "부정 증가"

        emotion_trends.append({"group": f"섭취 {index}번째", "trend": trend})

        for key in categorized_changes:
            total_scores[key] += categorized_changes[key]

        group_emotion.append({
            "group": f"섭취 {index}번째",
            "group_number": group,
            "positive": categorized_changes["positive"],
            "negative": categorized_changes["negative"],
            "neutral": categorized_changes["neutral"]
        })

    # ✅ 정규화
    total_sum = sum(total_scores.values())
    if total_sum > 0:
        total_scores = {k: round((v / total_sum) * 100, 2) for k, v in total_scores.items()}

    # ✅ 행동 분석
    df_behavior.columns = df_behavior.columns.str.strip()
    df_behavior_numeric = df_behavior.drop(columns=["Image File"])
    frame_behavior = df_behavior_numeric.groupby("Start Time (s)").mean().reset_index()

    behavior_total = {
        "positive_behavior": round(df_behavior["Positive (%)"].mean(), 2),
        "negative_behavior": round(df_behavior["Negative (%)"].mean(), 2)
    }

    # ✅ 가중치 적용 (감정 80% + 행동 20%)
    a, b = 0.8, 0.2
    Y_positive = a * total_scores["positive"] + b * behavior_total["positive_behavior"]
    Y_negative = a * total_scores["negative"] + b * behavior_total["negative_behavior"]
    Y_neutral = a * total_scores["neutral"]

    # ✅ JSON 데이터 생성
    summary_prompt = {
        "summary": {
            "감정 토탈": total_scores,
            "그룹별 감정 분석": group_emotion,
            "행동 토탈": behavior_total,
            "프레임별 행동 분석": frame_behavior.to_dict(orient="records"),
            "총합 감정+행동": {
                "Y_positive": round(Y_positive, 2),
                "Y_negative": round(Y_negative, 2),
                "Y_neutral": round(Y_neutral, 2)
            }
        },
        "analysis": {
            "emotion_trends": emotion_trends,
            "comment": "감정 변화와 행동 변화를 통합 분석하여 소비자 인사이트를 제공합니다."
        }
    }

    # ✅ 중요 키워드 감지 및 볼드 적용
    def extract_important_phrases(text):
        """GPT를 사용하여 중요 키워드를 JSON 리스트로 반환"""
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "당신은 텍스트 분석 전문가입니다.핵심이 될만한 단어들을 잘 찾아냅니다."},
                {"role": "user", "content": f"다음 텍스트에서 중요한 단어나 문구를 JSON 리스트로 반환하세요: {text}"}
            ],
            temperature=0.5
        )

        raw_response = response.choices[0].message.content.strip()
        try:
            important_phrases = json.loads(raw_response)
            if not isinstance(important_phrases, list):
                raise ValueError("JSON 리스트 아님")
        except (json.JSONDecodeError, ValueError):
            important_phrases = re.findall(r'"(.*?)"', raw_response)

        return [phrase for phrase in important_phrases if isinstance(phrase, str)]

    def apply_bold(text, important_phrases):
        """중요 키워드에 볼드 적용"""
        for phrase in important_phrases:
            text = re.sub(rf"\b{re.escape(phrase)}\b", f"<b>{phrase}</b>", text)
        return text

    # ✅ ChatGPT 분석 및 볼드 적용
    analysis_responses = {}

    for analysis_type, instruction in {
        "total_analysis": "햄버거 소비와 관련하여 감정 및 행동 분석을 바탕으로 판매 전략과 개선 방안을 제안하세요.",
        "emotion_analysis": "햄버거 소비와 관련하여 감정 변화만을 분석하고, 그룹별 감정 변화 패턴을 설명하세요.",
        "behavior_analysis": "햄버거 소비와 관련하여 행동 데이터만을 분석하고, 행동 변화 패턴을 설명하세요."
    }.items():
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "당신은 감정 및 행동 분석 전문가이며 번창한 햄버거집 사장님으로 판매전략에 있어 유능합니다."},
                {"role": "user", "content": json.dumps({"summary": summary_prompt["summary"], "analysis": instruction}, ensure_ascii=False)}
            ],
            temperature=0.7
        )

        original_text = response.choices[0].message.content
        important_phrases = extract_important_phrases(original_text)
        formatted_response = apply_bold(original_text, important_phrases)

        analysis_responses[analysis_type] = formatted_response

    # ✅ JSON 저장
    final_json = {"summary": summary_prompt["summary"], "analysis": analysis_responses}

    with open(output_json_path, "w", encoding="utf-8") as json_file:
        json.dump(final_json, json_file, ensure_ascii=False, indent=4)
        

def update_status(message):
    """진행 상태 메시지를 출력하는 함수"""
    status_data = {
        "type": "status",
        "message": message
    }
    print(json.dumps(status_data), flush=True)

def main():
        # 전체 작업 단계 정의 (11단계)
        update_status("분석을 시작합니다...")
        total_steps = 100
        progress_bar = tqdm(total=total_steps, desc="분석 진행중", unit="%")
        
        # 현재 스크립트 위치 (/home/jaeseok/jsworld/WeMeet/video-survey-node/scripts)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # video-survey-node 디렉토리 (상위 디렉토리)
        base_dir = os.path.dirname(script_dir)
        
       

        if len(sys.argv) > 1:
            video_folder = sys.argv[1]
            # 작업 디렉토리 변경
            os.chdir(video_folder)
            # 비디오 파일 찾기
            video_files = glob("*.mp4")
            if not video_files:
                print(json.dumps({"status": "error", "message": "No MP4 file found"}))
                sys.exit(1)
            video_path = video_files[0]
        else:
            raise ValueError("비디오 폴더 경로가 제공되지 않았습니다.")
        
        # 작업에 필요한 디렉토리 목록
        required_dirs = [
            os.path.join(video_folder, "mp4_to_image"),
            os.path.join(video_folder, "image_crop"), 
            os.path.join(video_folder, "predict"),
            os.path.join(video_folder, "final_output"),
            os.path.join(video_folder, "deepface_emotions")
        ]
        
        # 각 디렉토리 생성 
        for dir_path in required_dirs:
            os.makedirs(dir_path, exist_ok=True)

        # 1. 프레임 추출 (10%)
        extract_frames(video_path)
        update_status("1/11. 프레임 추출 완료")

        # 2. 손 기반 이미지 크롭 (10%)
        hand_processor = HandProcessor()
        hand_processor.process()
        update_status("2/11. 손 기반 이미지 크롭 완료")

        # 3. 이미지 분류 (10%)
        classifier = HamburgerClassifier()
        classifier.process()
        update_status("3/11. 이미지 분류 완료")

        # 4. 이미지 분석 (10%)
        detector = EatingActivityDetector()
        detector.process()
        update_status("4/11. 이미지 분석 완료")

        # 5. 얼굴과 손의 위치 감지 (10%)
        face_hand_processor = FaceHandProcessor()
        face_hand_processor.process()
        update_status("5/11. 얼굴과 손의 위치 감지 완료")

        # 6. 움직임 분석 (10%)
        movement_analyzer = MovementAnalyzer()
        positive_behave, negative_behave = movement_analyzer.process(video_path=video_path)
        update_status("6/11. 움직임 분석 완료")

        # 7. 감정 분석 (10%)
        emotion_analyzer = EmotionAnalyzer()
        emotion_analyzer.process()
        update_status("7/11. 감정 분석 완료")

        # 8. 감정 변화 분석 (10%)
        analyze_emotion_changes()
        update_status("8/11. 감정 변화 분석 완료")

        # 9. 감정 점수 계산 (10%)
        positive_emotion, negative_emotion, neutral_emotion = analyze_emotion_scores()
        update_status("9/11. 감정 점수 계산 완료")

        # 10. 최종 점수 계산 (5%)
        data = calculate_final_scores(
            positive_emotion=positive_emotion,
            negative_emotion=negative_emotion,
            neutral_emotion=neutral_emotion,
            positive_behave=positive_behave,
            negative_behave=negative_behave
        )
        update_status("10/11. 최종 점수 계산 완료")

        # 11. 보고서 생성 (5%)
        current_time = datetime.now().strftime("%y%m%d_%H%M%S")
        json_path = os.path.join(base_dir, "data", "analysis_results", f"{current_time}.json")
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        analyze_and_save_report(output_json_path=json_path)
        update_status("11/11. 보고서 생성 완료")

        progress_bar.close()

        return {
            "status": "success",
            "results_file": json_path,
            "message": "분석이 성공적으로 완료되었습니다."
        }



if __name__ == "__main__":
    if len(sys.argv) > 1:
        result = main()
        print(json.dumps(result))
        sys.stdout.flush()