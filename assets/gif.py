from moviepy import VideoFileClip

clip = VideoFileClip("/Users/ethan-zheng/Desktop/UMPE/assets/all_final_video.mp4")
clip.write_gif(
    "/Users/ethan-zheng/Desktop/UMPE/assets/all_final_video.gif",
    fps=5
)

