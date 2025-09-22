from moviepy import VideoFileClip 

input_path = "/Users/ethan-zheng/Desktop/UMPE/assets/mapping2.mp4"
output_path = "/Users/ethan-zheng/Desktop/UMPE/assets/mapping2.gif"

clip = VideoFileClip(input_path)

clip = clip.resized(0.5)

clip.write_gif(output_path)