
import cv2
import os
from multiprocessing import Pool
# import face_recognition
from PIL import Image
from moviepy.editor import VideoFileClip

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 从单个视频中抽取帧和音频
def extract_frames(video_data):
    emotion, video_file, emotion_dir, frame_emotion_dir = video_data
    video_path = os.path.join(emotion_dir, video_file)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    # 载入视频文件
    video_clip = VideoFileClip(video_path)
    fps = video_clip.fps  # 获取视频的帧率

    while True:
        success, frame = cap.read()
        if not success:
            break
            
        # 计算当前帧的时间戳
        frame_time = frame_count / fps

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 检测人脸
        face_locations = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
        for i, face_location in enumerate(face_locations):
            x, y, width, height = face_location 
            face_image = frame[y:(y+height), x:(x+width)]
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            face_image = Image.fromarray(face_image)

            # 保存图片
            frame_file = os.path.join(frame_emotion_dir, f"{video_file[:-4]}_frame_{frame_count:04d}.jpg")
            face_image.save(frame_file)
            
            # 提取并保存对应的音频片段
            audio_filename = os.path.join(frame_emotion_dir, f"{video_file[:-4]}_frame_{frame_count:04d}.wav")
            audio_clip = video_clip.subclip(frame_time, frame_time + 1.0 / fps)
            audio_clip.audio.write_audiofile(audio_filename, fps=44100)
        
        frame_count += 1

    cap.release()
    print(f"Extracted all frames from {video_file} into {frame_emotion_dir}")

def main():
    source_dir = 'VideoFlash'
    target_dir = 'VideoFrameAudio'

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Prepare a list of data for multiprocessing
    video_data_list = []

    for emotion in os.listdir(source_dir):
        emotion_dir = os.path.join(source_dir, emotion)
        if os.path.isdir(emotion_dir):
            frame_emotion_dir = os.path.join(target_dir, emotion)
            if not os.path.exists(frame_emotion_dir):
                os.makedirs(frame_emotion_dir)
                
            for video_file in os.listdir(emotion_dir):
                video_data_list.append((emotion, video_file, emotion_dir, frame_emotion_dir))

    # 创建多线程池
    pool = Pool(processes=4)
    pool.map(extract_frames, video_data_list)
    pool.close()
    pool.join()

    print("Frame extraction is complete.")

if __name__ == '__main__':
    main()