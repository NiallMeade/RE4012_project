import subprocess
import time
import sys
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput

# Define the Master resolution and the targets
master_res = (640, 480)
master_file = "capture_640x480.h264"

# The other resolutions you requested
targets = [
    {"res": "480", "name": "capture_480x480.mp4"},
    {"res": "320", "name": "capture_320x320.mp4"},
    {"res": "256", "name": "capture_256x256.mp4"}
]

picam2 = Picamera2()

def run_capture():
    # Configure for the highest resolution
    config = picam2.create_video_configuration(main={"size": master_res})
    picam2.configure(config)
    
    #picam2.set_controls({"Brightness": 0.25})
    
    print(f"--- Camera Ready ---")
    picam2.start()
    encoder = H264Encoder()
    picam2.start_recording(encoder, FileOutput(master_file))    
    try:
        print("RECORDING... Press Ctrl+C to stop manually.")
        while True:
            time.sleep(0.1) # Keep the script alive until interrupted
    except KeyboardInterrupt:
        print("\nStopping recording...")
        picam2.stop_recording()
        picam2.close()
        print("Master video saved.")

def resize_videos():
    print("\n--- Generating identical videos at different resolutions ---")
    for target in targets:
        print(f"Creating {target['name']}...")
        
        # ffmpeg command to scale the master video
        # -y: overwrite existing files
        # -vf scale: changes the resolution
        # -c:v libx264: uses a standard encoder for maximum compatibility
        cmd = [
            "ffmpeg", "-y", "-i", master_file,
            "-vf", f"crop=ih:ih,scale={target['res']}:{target['res']}",
            "-c:v", "libx264", "-preset", "slow", target['name']
        ]
        
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        print(f"Finished {target['res']}")

if __name__ == "__main__":
    run_capture()
    resize_videos()
    print("\nProcess complete. You now have 4 versions of the same footage.")
