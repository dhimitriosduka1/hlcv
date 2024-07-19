import cv2
import os
import datetime

def video_to_frames(video_path, output_folder, prefix='frame', format='jpg', frequency=1):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Get video properties
    fps = int(video.get(cv2.CAP_PROP_FPS))
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize frame counter
    count = 0

    while True:
        # Read a frame from the video
        success, frame = video.read()

        if not success:
            break

        # Save frame as an image if it meets the frequency condition
        if count % 5 == 0:
            frame_filename = f"{prefix}_{str(datetime.datetime.now())}_{count:04d}.{format}"
            frame_path = os.path.join(output_folder, frame_filename)
            cv2.imwrite(frame_path, frame)
            print(f"Saved {frame_filename}")

        count += 1

    # Release the video capture object
    video.release()

    print(f"Conversion complete. {count} frames processed.")

# Example usage
video_path = '/home/dhimitriosduka/Documents/UdS/SoSe 2024/High-Level Computer Vision/Assignments/hlcv/Screencast from 2024-07-17 03-24-24.mp4'
output_folder = '/home/dhimitriosduka/Documents/UdS/SoSe 2024/High-Level Computer Vision/Assignments/hlcv/photos'
video_to_frames(video_path, output_folder, frequency=0.3)