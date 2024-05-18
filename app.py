import streamlit as st
import torch
from PIL import Image
from io import BytesIO
import glob
from datetime import datetime
import os
import wget

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Configurations
CFG_MODEL_PATH = "models/best_weights_yolov5.pt"
CFG_ENABLE_URL_DOWNLOAD = True
if CFG_ENABLE_URL_DOWNLOAD:
    # Configure this if you set cfg_enable_url_download to True
    url = "https://archive.org/download/best_weights_yolov5/best_weights_yolov5.pt"

# End of Configurations

def imageInput(model, src):
    if src == 'Upload your own data.':
        image_file = st.file_uploader("Upload an Image of PNS X-ray", type=['png', 'jpeg', 'jpg'])
        # row1, row2 = st.columns(2)

        col1, col2 = st.columns(2)
        if image_file is not None:
            img = Image.open(image_file)
            st.image(img, caption='Uploaded Image', use_column_width = False, width=600)

            # st.image(img, caption='Uploaded Image', use_column_width='always')
            # with col1:
            #     st.image(img, caption='Uploaded Image', use_column_width='always')
           
            
            ts = datetime.timestamp(datetime.now())
            imgpath = os.path.join('data/uploads', str(ts) + image_file.name)
            outputpath = os.path.join('data/outputs', os.path.basename(imgpath))
            with open(imgpath, mode="wb") as f:
                f.write(image_file.getbuffer())

            with st.spinner(text="Predicting..."):
                pred = model(imgpath)
                pred.render()
                for im in pred.ims:
                    im_base64 = Image.fromarray(im)
                    im_base64.save(outputpath)

            img_ = Image.open(outputpath)
            st.image(img_, caption='Model Prediction(s)', use_column_width= False, width=600)

            # st.image(img_, caption='Model Prediction(s)', use_column_width='always')
            # with col2:
            #     st.image(img_, caption='Model Prediction(s)', use_column_width='always')


def main():
    if CFG_ENABLE_URL_DOWNLOAD:
        downloadModel()
    else:
        if not os.path.exists(CFG_MODEL_PATH):
            st.error('Model not found, please config if you wish to download model from url set `cfg_enable_url_download = True`', icon="⚠️")

    st.sidebar.title('Computation Device')
    
    datasrc = 'Upload your own data.'  # Only allow upload option
    option = 'Image'  # Only allow image option

    if torch.cuda.is_available():
        deviceoption = st.sidebar.radio("Select compute Device.", ['cpu', 'cuda'], disabled=True, index=1)
    else:
        deviceoption = st.sidebar.radio("Select compute Device.", ['cpu', 'cuda'], disabled=False, index=0)


    # adding images here 
    # import streamlit as st
    # from PIL import Image

    # Load your X-ray image
    xray_image = Image.open("./Emoji/up.jpeg")

# Display the image and the header
    st.image(xray_image, width=50)  # Adjust the width as needed
    st.header('PNS detection using X-rays')




    # st.header('📦 PNS detection using X-rays')
    st.sidebar.markdown("We are pleased to announce the development of a new system that utilizes X-ray images to identify signs of sinusitis. This innovative tool can streamline the diagnostic process by offering a fast and accurate analysis. By providing healthcare professionals with a helpful indicator, it can ultimately lead to improved patient care.")

    imageInput(loadmodel(deviceoption), datasrc)
    
@st.cache_resource
def downloadModel():
    if not os.path.exists(CFG_MODEL_PATH):
        wget.download(url, out="models/")

@st.cache_resource
def loadmodel(device):
    if CFG_ENABLE_URL_DOWNLOAD:
        model_path = f"models/{url.split('/')[-1]}"
    else:
        model_path = CFG_MODEL_PATH
    custom = "--hide-conf"
    model = torch.hub.load('ultralytics/yolov5', "custom" , path=model_path, force_reload=True, device=device)
    return model

if __name__ == '__main__':
    main()
