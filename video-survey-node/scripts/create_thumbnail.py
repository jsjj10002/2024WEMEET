import cv2
import sys
import os

def create_thumbnail(video_path, output_path):
    try:
        # 비디오 파일 열기
        cap = cv2.VideoCapture(video_path)
        
        # 비디오가 제대로 열렸는지 확인
        if not cap.isOpened():
            print(f"Error: Could not open video file: {video_path}")
            return False
            
        # 1초 위치로 이동 (프레임 레이트 * 1초)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
        
        # 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            return False
            
        # 썸네일 크기 조정 (예: 320x240)
        thumbnail = cv2.resize(frame, (320, 240))
        
        # 썸네일 저장
        cv2.imwrite(output_path, thumbnail)
        
        # 자원 해제
        cap.release()
        
        return True
        
    except Exception as e:
        print(f"Error creating thumbnail: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python create_thumbnail.py <video_path> <output_path>")
        sys.exit(1)
        
    video_path = sys.argv[1]
    output_path = sys.argv[2]
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
        
    if create_thumbnail(video_path, output_path):
        print(f"Thumbnail created successfully: {output_path}")
    else:
        print("Failed to create thumbnail")
        sys.exit(1)