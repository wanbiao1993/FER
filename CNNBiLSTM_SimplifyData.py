import os
import shutil
import math
import sys
from random import sample

def copy_fraction_of_videos(source_dir, target_dir, fraction):
    # 确保目标目录存在
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 遍历VideoFlash目录中的每个子目录
    for root, dirs, files in os.walk(source_dir):
        for dir_name in dirs:
            # 获取每个类别的完整路径
            category_path = os.path.join(root, dir_name)
            
            # 获取所有视频文件，这里假设视频文件的扩展名为.mp4，根据需要修改
            video_files = [f for f in os.listdir(category_path) if f.endswith('.flv')]
            
            # 计算需要复制的视频文件数量
            num_files_to_copy = math.ceil(len(video_files) / fraction)
            
            # 随机选择所需比例的视频文件
            selected_files = sample(video_files, min(num_files_to_copy, len(video_files)))
            
            # 创建相应的目标目录
            target_category_path = os.path.join(target_dir, dir_name)
            if not os.path.exists(target_category_path):
                os.makedirs(target_category_path)
            
            # 复制选定的视频文件
            for file_name in selected_files:
                source_file_path = os.path.join(category_path, file_name)
                target_file_path = os.path.join(target_category_path, file_name)
                shutil.copy2(source_file_path, target_file_path)
                print(f'Copied {source_file_path} to {target_file_path}')

    print('Finished copying files.')

if __name__ == "__main__":
    # 获取命令行参数
    if len(sys.argv) != 4:
        print("Usage: python CNNBiLSTM_Simplify.py <source_dir> <target_dir> <fraction>")
        sys.exit(1)

    source_dir = sys.argv[1]
    target_dir = sys.argv[2]
    fraction = int(sys.argv[3])

    copy_fraction_of_videos(source_dir, target_dir, fraction)