import os
import shutil

# 视频文件的目录
video_dir = 'VideoFlash'
# 支持的视频文件扩展名
video_ext = '.flv'
# 情绪到目录名的映射
emotions = {
    'ANG': 'Anger',
    'DIS': 'Disgust',
    'FEA': 'Fear',
    'HAP': 'Happy',
    'NEU': 'Neutral',
    'SAD': 'Sad'
}

# 确保VideoFlash目录存在
if not os.path.exists(video_dir):
    print(f"Error: The directory '{video_dir}' does not exist.")
else:
    # 遍历VideoFlash目录中的所有文件
    for filename in os.listdir(video_dir):
        # 检查扩展名是否为.flv
        if filename.lower().endswith(video_ext):
            try:
                # 解析文件名以获取情绪
                parts = filename.split('_')
                emotion_code = parts[2]
                
                # 获取情绪的完整名称
                emotion_full = emotions.get(emotion_code.upper(), None)
                
                if emotion_full:
                    # 创建目标目录如果它还不存在
                    target_dir = os.path.join(video_dir, emotion_full)
                    if not os.path.exists(target_dir):
                        os.makedirs(target_dir)
                    
                    # 移动文件到新目录
                    src_path = os.path.join(video_dir, filename)
                    dst_path = os.path.join(target_dir, filename)
                    shutil.move(src_path, dst_path)
                    print(f"Moved '{filename}' to '{emotion_full}' folder.")
                else:
                    print(f"Warning: The file '{filename}' does not match the expected pattern for emotions.")
            except IndexError:
                print(f"Error: The file '{filename}' does not match the expected naming convention.")
            except Exception as e:
                print(f"An error occurred: {e}")