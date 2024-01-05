import librosa
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor

def calculate_mfcc(audio_file):
    # 加载音频文件
    y, sr = librosa.load(audio_file, sr=None)
    # 计算MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    print(mfcc.shape)
    return mfcc

def process_batch(batch_files, audio_folder,label):
    # 存储当前批次的MFCCs
    batch_mfcc = []

    # 遍历当前批次的文件
    for file in batch_files:
        # 计算当前音频文件的MFCC
        mfcc = calculate_mfcc(os.path.join(audio_folder, file))
        # 添加到批次MFCCs列表
        batch_mfcc.append(mfcc)
    
    return np.array(batch_mfcc),label

def process_audio_files(audio_folder,class_name):
    # 存储最终的MFCC数组
    mfcc_arrays = []
    labels = []

    # 获取所有音频文件的路径
    audio_files = [f for f in os.listdir(audio_folder) if f.endswith('.wav')]
    audio_files.sort()
    
    # 创建线程池
    with ThreadPoolExecutor() as executor:
        # 计划任务
        futures = []
        for i in range(0, len(audio_files), 10):
            # 获取当前批次的音频文件
            batch_files = audio_files[i:i+10]
            if len(batch_files) != 10:
                print(f"ignore {i}")
                continue
            # 提交当前批次的任务到线程池
            futures.append(executor.submit(process_batch, batch_files, audio_folder,class_name))
            # print(f"{batch_files},labesl:{class_name}")

        # 等待所有线程完成并收集结果
        for future in futures:
            mfcc_result, label_result = future.result()
            mfcc_arrays.append(mfcc_result)
            labels.append(label_result)
    
    return mfcc_arrays,labels

def save_mfcc_data(mfcc_data,labels, class_name, output_folder):
    # 确保输出文件夹存在
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    
    # 构建保存MFCC数组数据的文件路径
    output_path = os.path.join(output_folder, f"{class_name}_mfcc.npy")
    
    labels_output_path = os.path.join(output_folder, f"{class_name}_labels.npy")
    
    # 保存MFCC数组到文件
    np.save(output_path, mfcc_data)
    np.save(labels_output_path, labels)

def main():
    # 定义遍历的根目录
    root_dir = 'VideoFrameAudio'
    
    # 存储最终的输出目录
    output_dir = 'AudioFrameNpy'

    # 创建线程池
    with ThreadPoolExecutor() as executor:
        # 计划任务
        futures = []
        # 遍历root_dir下的所有子目录
        for class_name in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_name)
            
            # 检查是否为目录
            if not os.path.isdir(class_path):
                continue
            
            # 提交子目录处理的任务到线程池
            futures.append(executor.submit(process_audio_files, class_path,class_name))

        # 在主线程中处理每个任务的结果
        for future in futures:
            # print(f"future {(future.result())} len {len(future.result()[0])}")
            class_name = os.path.basename(future.result()[1][0])
            mfcc_data,labels = future.result()
            # print(mfcc_data)
            save_mfcc_data(mfcc_data, labels,class_name, output_dir)
            print(f"Processed and saved MFCC data for class: {class_name}")

if __name__ == '__main__':
    main()