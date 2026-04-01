# 把原始图片转换为低分辨率并保存，用于推理。
import cv2
import os

input_dir = "/Users/leon.w/workspace/learn/FaceMe/test/ref"
output_dir = "/Users/leon.w/workspace/learn/FaceMe/test/lq"

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

target_size = 256  # 目标短边长度

for filename in os.listdir(input_dir):
    filepath = os.path.join(input_dir, filename)
    
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        img = cv2.imread(filepath)
        
        if img is not None:
            # 计算保持比例的新尺寸
            h, w = img.shape[:2]
            if h > w:
                new_w = target_size
                new_h = int(h * (target_size / w))
            else:
                new_h = target_size
                new_w = int(w * (target_size / h))
            
            # 调整大小
            img = cv2.resize(img, (new_w, new_h))
            
            # 高斯模糊
            img = cv2.GaussianBlur(img, (9, 9), 0)
            
            # 保存为原始格式
            output_path = os.path.join(output_dir, filename)
            
            # 根据文件扩展名选择保存参数
            ext = os.path.splitext(filename)[1].lower()
            if ext in ['.jpg', '.jpeg']:
                # JPEG 格式，使用质量参数
                cv2.imwrite(output_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 30])
            elif ext == '.png':
                # PNG 格式，使用压缩参数
                cv2.imwrite(output_path, img, [int(cv2.IMWRITE_PNG_COMPRESSION), 6])
            else:
                # 其他格式，直接保存
                cv2.imwrite(output_path, img)
            
            print(f"Processed: {filename}")
        else:
            print(f"Failed to read: {filename}")

print("\nAll images processed!")