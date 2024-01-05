import os
import pickle
from PIL import Image
import numpy as np
import threading
import math

# 情绪和ID的映射字典，用于将文本标签映射到数字
emotion_to_id = {
    'Anger': 0,
    'Disgust': 1,
    'Fear': 2,
    'Happy': 3,
    'Neutral': 4,
    'Sad': 5,
    # ... 添加其他情绪和ID的映射 ...
}

# 线程锁，用于在写入文件时防止竞态条件
lock = threading.Lock()

def process_emotion_images(emotion, emotion_path, emotion_id):
    """
    处理单个情绪文件夹下的所有图片，将它们转换为numpy数组，并保存到pickle文件中。
    
    参数:
    emotion: 情绪的名称（如'Happy'）
    emotion_path: 包含情绪图片的文件夹路径
    emotion_id: 与情绪对应的唯一标识符
    """
    x = [] # 临时存储总数据
    y = [] # 临时存储总数据
    img_n = []  # 临时存储图片数组
    batch_size = 10  # 每组图片的数量
    pices = 1 # 分成多少片

    # 遍历情绪文件夹中的所有图片
    files = os.listdir(emotion_path)
    image_files = sorted([f for f in files if f.endswith('.jpg')])
    image_files.sort()
    # 如果不排序，可能读到不是同一个视频中的帧序列
    files = image_files
    # for debug
    # with open('filename.txt', 'w') as f:
    #     # 遍历文件名列表，并将每个文件名写入文件
    #     for file_name in files:
    #         f.write("%s\n" % file_name)
    for img_file in files:
        img_path = os.path.join(emotion_path, img_file)
        with Image.open(img_path) as img:
            img = img.resize((50, 50))  # 调整图片大小,太大GPU资源不够
            img_array = np.array(img)  # 将图片转换为numpy数组
            img_n.append(img_array)  # 将图片数组添加到列表中
            

        # 当积累到10个图片时，保存到pickle文件
        if len(img_n) == batch_size:
            x.append(img_n)
            y.append(emotion_id) 

            # 清空列表，准备下一批次的数据
            img_n = []
    
    # 如果有图片，则将它们保存到pickle文件
    if x:
        with lock:
            # 保存图片数据,分成三组
            # 确定每部分的大小。向下取整确保每份至少有 equal_portion_size 个元素
            target_dir = 'VideoFramePickle-Simple'

            if not os.path.exists(target_dir):
                os.makedirs(target_dir)

            equal_portion_size_x = math.floor(len(x) / pices)
            
            if pices == 1:
                
                x_last = x
                pickle_file_last = os.path.join(target_dir, f'{emotion}_x_{pices}.pickle') 
                with open(pickle_file_last, 'wb') as f_last:
                    pickle.dump(x_last, f_last) 

                y_last = y
                pickle_file_last = os.path.join(target_dir, f'{emotion}_y_{pices}.pickle') 
                with open(pickle_file_last, 'wb') as f_last:
                    pickle.dump(y_last, f_last) 
            else:

                for pice in range(pices-1):
                    x_tmp = x[pice*equal_portion_size_x:(pice+1)*equal_portion_size_x]
                    pickle_file = os.path.join(target_dir, f'{emotion}_x_{pice+1}.pickle') 
                    with open(pickle_file, 'wb') as f_tmp:
                        pickle.dump(x_tmp, f_tmp) 

                x_last = x[(pices-1)*equal_portion_size_x:]
                pickle_file_last = os.path.join(target_dir, f'{emotion}_x_{pices}.pickle') 
                with open(pickle_file_last, 'wb') as f_last:
                    pickle.dump(x_last, f_last) 

        
                # 保存对应的标签
                equal_portion_size_y = math.floor(len(y) / pices)
                for pice in range(pices-1):
                    y_tmp = y[pice*equal_portion_size_y:(pice+1)*equal_portion_size_y]
                    pickle_file = os.path.join(target_dir, f'{emotion}_y_{pice+1}.pickle') 
                    with open(pickle_file, 'wb') as f_tmp:
                        pickle.dump(y_tmp, f_tmp) 

                y_last = y[(pices-1)*equal_portion_size_y:]
                pickle_file_last = os.path.join(target_dir, f'{emotion}_y_{pices}.pickle') 
                with open(pickle_file_last, 'wb') as f_last:
                    pickle.dump(y_last, f_last) 

        print(f"Data for emotion '{emotion}' has been pickled.")
    else:
        print(f"No images found for emotion '{emotion}'.")  # 如果没有图片，输出警告

def read_data(directory):
    """
    读取给定目录下的所有情绪文件夹，并为每个情绪启动一个处理线程。
    
    参数:
    directory: 包含情绪子文件夹的主目录
    """
    threads = []  # 存储所有线程的列表

    # 遍历主目录中的每个子目录
    for emotion in os.listdir(directory):
        emotion_path = os.path.join(directory, emotion)  # 获取情绪文件夹的完整路径
        print(f"Start processing {emotion}")
        
        # 检查路径是否为目录
        if os.path.isdir(emotion_path):
            emotion_id = emotion_to_id.get(emotion)  # 获取情绪对应的ID
            if emotion_id is None:
                print(f"Emotion '{emotion}' is not recognized and will be skipped.")
                continue  # 如果情绪不在映射字典中，跳过

            # 创建新线程处理该情绪的图片
            thread = threading.Thread(target=process_emotion_images, args=(emotion, emotion_path, emotion_id))
            threads.append(thread)  # 将线程添加到列表中
            thread.start()  # 启动线程
    
    # 等待所有线程完成
    for thread in threads:
        thread.join()

# 读取VideoFrame目录下的数据
# read_data('VideoFrame-Face')
# read_data('AsiaFaceFrame')
read_data('VideoFrameAudio-Simple')

print("All datasets have been successfully pickled into emotion-specific files.")