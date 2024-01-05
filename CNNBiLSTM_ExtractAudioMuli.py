import moviepy.editor as mp
import librosa
import numpy as np
import os

def extract_mfcc_from_video(video_path, n_mfcc=40):
    """
    从视频文件中提取音频，并计算与视频帧数匹配的MFCC特征。
    如果最后一帧音频长度不够，则舍弃该帧。

    参数:
    video_path (str): 视频文件的路径。
    n_mfcc (int): 每个MFCC矢量的组成部分数量。

    返回:
    numpy.ndarray: MFCC特征矩阵。
    """

    # 从视频中提取音频信号
    video = mp.VideoFileClip(video_path)

    # 提取音频
    audio = video.audio
    audio_path = f'{video_path}_temp_audio.wav'  # 临时存储音频的路径
    audio.write_audiofile(audio_path)

    # 获取视频的帧率
    frame_rate = video.fps  # 帧每秒

    # 获取音频的采样率
    audio_sample_rate = audio.fps  # 音频样本每秒

    # 计算每个视频帧对应的音频样本数
    samples_per_frame = int(audio_sample_rate / frame_rate)

    # 加载音频并计算MFCC
    y, sr = librosa.load(audio_path, sr=audio_sample_rate)

    # 计算总帧数，舍弃最后不完整的帧
    total_frames = int(len(y) / samples_per_frame)
    print(f"path {video_path} frames {total_frames}")

    # 确定hop_length为每个视频帧对应的音频样本数
    hop_length = samples_per_frame

    # 计算MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)

    # 如果mfcc的帧数多于视频帧数，舍弃多余的
    if mfcc.shape[1] > total_frames:
        mfcc = mfcc[:, :total_frames]

    # 清除临时音频文件
    os.remove(audio_path)

    return mfcc

import os
import concurrent.futures
from moviepy.editor import VideoFileClip
import librosa
import numpy as np

def process_video_file(video_path):
    """
    处理单个视频文件并返回其 MFCC
    """
    try:
        return extract_mfcc_from_video(video_path)
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return None

def process_videos_concurrently(root_dir, max_workers=4):
    """
    使用多线程在 root_dir 下的所有子目录中并发处理视频文件。

    参数:
    root_dir (str): 包含视频类别子目录的根目录路径。
    max_workers (int): 线程池中的最大线程数。

    返回:
    dict: 一个字典，键为类别名称，值为该类别下所有视频的 MFCC 列表。
    """
    category_mfccs = {}
    output_dir = 'VideoFrameAudio-MFCC'

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 为每个视频文件创建一个 future 列表
        future_to_video = {}

        for subdir, dirs, files in os.walk(root_dir):
            for category in dirs:
                category_path = os.path.join(subdir, category)
                category_mfccs[category] = []
                files = os.listdir(category_path)
                files.sort()
                print(files)
                for file in files:
                    if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv','.flv')):
                        video_path = os.path.join(category_path, file)
                        # 安排线程池中的线程来处理视频
                        future = executor.submit(process_video_file, video_path)
                        future_to_video[future] = (category, video_path)

        # 处理完成的 futures
        for future in concurrent.futures.as_completed(future_to_video):
            category, video_path = future_to_video[future]
            target_dir = f"{output_dir}/{category}"
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            target_path = f"{target_dir}/{os.path.splitext(os.path.basename(video_path))[0]}_MFCC"
            print(f"category {category} path {target_path}")
            try:
                mfcc = future.result()
                if mfcc is not None:
                    category_mfccs[category].append(mfcc)
                    np.save(target_path, mfcc)
                    
            except Exception as e:
                print(f"Error processing {video_path}: {e}")

    return category_mfccs

# 设置 VideoFlash 目录的路径
root_directory = 'VideoFlash'
# 并发处理视频并获取结果
results = process_videos_concurrently(root_directory, max_workers=100)  # 可以根据您的系统调整线程数
print("all done")