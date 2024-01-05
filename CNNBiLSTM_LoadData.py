# 加载数据
import os
import numpy as np
from PIL import Image
import librosa

# 加载数据
def load_data(directory, batch_size=10,width=50,height=50,mfcc_dim = 128,process_audio=True,max_files = -1):
    emotion_to_id = {
        'Anger': 0,
        'Disgust': 1,
        'Fear': 2,
        'Happy': 3,
        'Neutral': 4,
        'Sad': 5,
        # ... 添加其他情绪和ID的映射 ...
    }
    categories = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    data = []  # to store batches of images and audio
    categories.sort()
    print(f"categories {categories}")

    for category in categories:
        category_path = os.path.join(directory, category)
        files = os.listdir(category_path)

        image_files = sorted([f for f in files if f.endswith('.jpg')])
        image_files.sort()
        audio_files = sorted([f for f in files if f.endswith('.wav')])
        audio_files.sort()
        print(f"start process {category} all imagefiles {len(image_files)} all audiofiles {len(audio_files)}")
        
        img_n = [] # 10个一组
        mfcc_features = []
        labels = []
        for index,img in enumerate(image_files):
            if max_files > 0 and index > max_files:
                print(f"arrvial max_files {max_files} index {index}")
                break;
                
            imagename = img
            #提取图片
            img_path = os.path.join(category_path, img)
            with Image.open(img_path) as img:
                img = img.resize((width, height))  # 调整图片大小,太大GPU资源不够
                img_array = np.array(img)  # 将图片转换为numpy数组
                img_n.append(img_array)  # 将图片数组添加到列表中
            
            if process_audio :
                #提取音频
                aud = audio_files[index]
                # 判断两个文件名是否相等
                image_name, ext_1 = os.path.splitext(imagename)
                audio_name, ext_2 = os.path.splitext(aud)
                if image_name != audio_name:
                    raise ValueError(f"{image_name} does not equal to {audio_name}")

                # 提取每个音频文件的MFCC特征

                # 加载音频文件，保留其原始采样率
                audio, sampling_rate = librosa.load(os.path.join(category_path, aud), sr=None, mono=True)
                # 计算MFCC特征
                mfcc = librosa.feature.mfcc(y=audio, sr=sampling_rate, n_mfcc=mfcc_dim,n_fft=len(audio), hop_length=len(audio))
                # 标准化
                mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
                mfcc = mfcc.T
                mfcc_features.append(mfcc)
            
            if len(img_n) == batch_size:
                data.append({'category': emotion_to_id[category],'images': img_n,'audio': mfcc_features})
                # 清空列表，准备下一批次的数据
                img_n = []
                mfcc_features = []
                labels = []
    print("process all files")
    return data

# 绘制历史数据
def draw_history(history,epoch):
    import matplotlib.pyplot as plt
    epochs = list(range(epoch))
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    plt.plot(epochs, acc, label='train accuracy')
    plt.plot(epochs, val_acc, label='val accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.plot(epochs, loss, label='train loss')
    plt.plot(epochs, val_loss, label='val loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
    return