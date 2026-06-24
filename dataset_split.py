"""
数据集划分 - 将源目录所有图片合并后按70-15-15划分为训练集、验证集和测试集
"""
import os
import shutil
import random
from glob import glob

def split_dataset(source_dir, train_ratio=0.7, val_ratio=0.15):
    # 输出目录设置
    base_output_dir = 'skin diseases'
    train_dir = os.path.join(base_output_dir, 'train-new')
    val_dir = os.path.join(base_output_dir, 'val')
    test_dir = os.path.join(base_output_dir, 'test')

    # 创建输出目录
    for directory in [train_dir, val_dir, test_dir]:
        os.makedirs(directory, exist_ok=True)

    # 递归收集所有图片并保留类别信息
    images = []
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                class_name = os.path.basename(root)
                images.append((class_name, file, os.path.join(root, file)))

    # 打乱数据
    random.seed(42)
    random.shuffle(images)

    # 计算划分数量
    train_count = int(len(images) * train_ratio)
    val_count = int(len(images) * val_ratio)
    test_count = len(images) - train_count - val_count

    # 划分数据集
    train_images = images[:train_count]
    val_images = images[train_count:train_count + val_count]
    test_images = images[train_count + val_count:]

    # 复制文件到对应目录
    def copy_files(images, target_dir):
        for class_name, file, src_path in images:
            class_dir = os.path.join(target_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            shutil.copy(src_path, os.path.join(class_dir, file))

    copy_files(train_images, train_dir)
    copy_files(val_images, val_dir)
    copy_files(test_images, test_dir)

    # 打印统计信息
    print(f"总计图片: {len(images)}")
    print(f"训练集: {len(train_images)} ({len(train_images)/len(images)*100:.1f}%)")
    print(f"验证集: {len(val_images)} ({len(val_images)/len(images)*100:.1f}%)")
    print(f"测试集: {len(test_images)} ({len(test_images)/len(images)*100:.1f}%)")
    print(f"\n训练集目录: {train_dir}")
    print(f"验证集目录: {val_dir}")
    print(f"测试集目录: {test_dir}")

if __name__ == '__main__':
    # 修改为你的源目录路径
    split_dataset('data')