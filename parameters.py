# Camera Dictionary
IP_CAMERAS = {
    "cam_1": [0, (640, 360, 0, 0)]
    # "cam_1": ["http://192.168.21.115:5000", (320, 200, -100, 0)],
    # "cam_2": ["http://192.168.21.115:5050", (320, 200, -50, 0)],
    # "cam_3": ["http://192.168.21.115:5020", (320, 200, 0, 0)],
    # "cam_4": ["http://192.168.21.115:5030", (320, 200, 50, 0)],
    # "cam_5": ["http://192.168.21.115:5040", (320, 200, 100, 0)]
}

# Frame Dimensions
FRAME_WIDTH = 640
FRAME_HEIGHT = 360

# Buffer Length
BUFFER_SIZE = 2048

# Log File Path
LOG_FILE_PATH = "logs/multicam_server.log"

# Camera reinitialization time (30 seconds)
IP_CAM_REINIT_WAIT_DURATION = 30

# Batch Size
BATCH_SIZE = 16
