from PIL import Image
import sys
import os

# 获取输入参数
if len(sys.argv) < 2:
    print("用法: python gif_reducer.py 输入.gif [输出.gif] [帧率]")
    sys.exit(1)

input_gif = sys.argv[1]
output_gif = sys.argv[2] if len(sys.argv) > 2 else input_gif.replace('.gif', '_reduced.gif')
target_fps = int(sys.argv[3]) if len(sys.argv) > 3 else 10

# 打开GIF并提取所有帧
with Image.open(input_gif) as img:
    frames = []
    try:
        while True:
            frames.append(img.copy())
            img.seek(len(frames))  # 移动到下一帧
    except EOFError:
        pass  # 已读取所有帧

# 计算跳帧间隔（每n帧保留1帧）
original_fps = 24  # 假设原始为24fps，可以调整
skip_ratio = max(1, original_fps // target_fps)

# 选择要保留的帧
selected_frames = frames[::skip_ratio]

# 保存为新GIF
if selected_frames:
    selected_frames[0].save(
        output_gif,
        save_all=True,
        append_images=selected_frames[1:],
        duration=1000//target_fps,  # 每帧持续时间(ms)
        loop=0  # 无限循环
    )
    print(f"完成! 已保存为: {output_gif}")
    print(f"帧数从 {len(frames)} 减少到 {len(selected_frames)}")
else:
    print("错误: 没有帧可保存")