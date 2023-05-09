import streamlit as st
import cv2
import torch
from utils.hubconf import custom
import numpy as np
import tempfile
import time
from collections import Counter
import json
import pandas as pd
from model_utils import get_yolo, color_picker_fn, get_system_stat
from ultralytics import YOLO

with open("/Users/sriramapadmaprabha/Documents/T-Systems/videoanalyticv2/design.css") as source_des:
    st.markdown(f"<style>{source_des.read()}</style>",unsafe_allow_html=True)

new_title = '<p style="font-family:sans-serif; color:#e20074; font-size: 40px;">T-Video Analytcs</p>'
st.markdown(new_title, unsafe_allow_html=True)

hide_st_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style
"""
st.markdown(hide_st_style, unsafe_allow_html=True) 


p_time = 0

st.sidebar.title('Settings')
# Choose the model
model_type = st.sidebar.selectbox(
    'Choose  Model', ('Base Model', 'Modelv8', 'Modelv7')
)

sample_img = cv2.imread('T-logo.jpg')
FRAME_WINDOW = st.image(sample_img, channels='BGR')
cap = None

if not model_type == 'Base Model':
    path_model_file = st.sidebar.text_input(
        f'path to {model_type} Model:',
        f'/Users/sriramapadmaprabha/Documents/T-Systems/videoanalyticv2/yolov8.pt'
    )
    if st.sidebar.checkbox('Load Model'):
        
        # YOLOv7 Model
        if model_type == 'Modelv7':
            # GPU
            gpu_option = st.sidebar.radio(
                'PU Options:', ('CPU', 'GPU'))

            if not torch.cuda.is_available():
                st.sidebar.warning('CUDA Not Available, So choose CPU', icon="⚠️")
            else:
                st.sidebar.success(
                    'GPU is Available on this Device, Choose GPU for the best performance',
                    icon="✅"
                )
            # Model
            if gpu_option == 'CPU':
                model = custom(path_or_model=path_model_file)
            if gpu_option == 'GPU':
                model = custom(path_or_model=path_model_file, gpu=True)

        # YOLOv8 Model
        if model_type == 'Modelv8':
            model = YOLO(path_model_file)

        

        # Inference Mode
        options = st.sidebar.radio(
            'Options:', ('Webcam', 'Video', 'RTSP'), index=1)

        # Confidence
        confidence = st.sidebar.slider(
            'Detection Confidence', min_value=0.0, max_value=1.0, value=0.25)

        # Draw thickness
        draw_thick = st.sidebar.slider(
            'Draw Thickness:', min_value=1,
            max_value=20, value=3
        )
        
        

        # Image
        
        # if options == 'Image':
        #     upload_img_file = st.sidebar.file_uploader(
        #         'Upload Image', type=['jpg', 'jpeg', 'png'])
        #     if upload_img_file is not None:
        #         pred = st.checkbox(f'Predict Using {model_type}')
        #         file_bytes = np.asarray(
        #             bytearray(upload_img_file.read()), dtype=np.uint8)
        #         img = cv2.imdecode(file_bytes, 1)
        #         FRAME_WINDOW.image(img, channels='BGR')

        #         if pred:
        #             img, current_no_class = get_yolo(img, model_type, model, confidence, color_pick_list, class_labels, draw_thick)
        #             FRAME_WINDOW.image(img, channels='BGR')

        #             # Current number of classes
        #             class_fq = dict(Counter(i for sub in current_no_class for i in set(sub)))
        #             class_fq = json.dumps(class_fq, indent = 4)
        #             class_fq = json.loads(class_fq)
        #             df_fq = pd.DataFrame(class_fq.items(), columns=['Class', 'Number'])
                    
        #             # Updating Inference results
        #             with st.container():
        #                 st.markdown("<h2>Inference Statistics</h2>", unsafe_allow_html=True)
        #                 st.markdown("<h3>Detected objects in curret Frame</h3>", unsafe_allow_html=True)
        #                 st.dataframe(df_fq, use_container_width=True)
        
        # Video
        if options == 'Video':
            upload_video_file = st.sidebar.file_uploader(
                'Upload Video', type=['mp4', 'avi', 'mkv'])
            if upload_video_file is not None:
                pred = st.checkbox(f'Predict Using {model_type}')

                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(upload_video_file.read())
                cap = cv2.VideoCapture(tfile.name)
                # if pred:


        # Web-cam
        if options == 'Webcam':
            cam_options = st.sidebar.selectbox('Webcam Channel',
                                            ('Select Channel', '0', '1', '2', '3'))
        
            if not cam_options == 'Select Channel':
                pred = st.checkbox(f'Predict Using {model_type}')
                cap = cv2.VideoCapture(int(cam_options))


        # RTSP
        if options == 'RTSP':
            rtsp_url = st.sidebar.text_input(
                'RTSP URL:',
                'rtsp://TestDevice:Tssg1234@192.168.50.147:554/stream2'
            )
            pred = st.checkbox(f'Predict Using {model_type}')
            cap = cv2.VideoCapture(rtsp_url)
    # Load Class names
        class_labels = model.names
        # color_pick_list = []
        # for i in range(len(class_labels)):
        #     classname = class_labels[i]
        #     color = color_picker_fn(classname, i)
        #     color_pick_list.append(color)


if (cap != None) and pred:
    stframe1 = st.empty()
    stframe2 = st.empty()
    stframe3 = st.empty()
    while True:
        success, img = cap.read()
        if not success:
            st.error(
                f"{options} NOT working\nCheck {options} properly!!",
                icon="🚨"
            )
            break

        img, current_no_class = get_yolo(img, model_type, model, confidence, class_labels, draw_thick)
        FRAME_WINDOW.image(img, channels='BGR')

        # FPS
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time
        
        # Current number of classes
        class_fq = dict(Counter(i for sub in current_no_class for i in set(sub)))
        class_fq = json.dumps(class_fq, indent = 4)
        class_fq = json.loads(class_fq)
        df_fq = pd.DataFrame(class_fq.items(), columns=['Class', 'Number'])
        print("values============================",)
        
        # Updating Inference results
        get_system_stat(stframe1, stframe2, stframe3, fps, df_fq)

