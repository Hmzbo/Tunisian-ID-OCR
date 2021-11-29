import streamlit as st

import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

import tensorflow as tf
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing import image

import torch
from arabicocr import arabic_ocr

from MyUtils import preprocess, clean_sort_results

# The following line of code make TensorFlow run on CPU
# Warining: Running TF on GPU will cause Yolo exceution to raise an error due to conflict of using gpu by both TF and Torch
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Loading CIN classification pretrained model
def load_effnet_model():
    effnet_model = load_model('./EffNetB0_CIN.h5')
    return effnet_model

# Funtion that interprets prediction results
def is_cin(pred):
    return True if np.round(pred) == 0 else False


def show_cin_ocr_page():
    # Upload CIN image
    st.header('Tunisian ID card OCR service')
    files = st.file_uploader('Upload ID card image(s)', accept_multiple_files=True, type=['png','jpeg','jpg'])
    if files != None:
        mobnet_model = load_effnet_model()
        imgs_num = len(files)
        if imgs_num == 1:
            image = Image.open(files[0])
            prepro_img = np.expand_dims(np.array(image.resize((224,224))),axis=0)
            pred = mobnet_model.predict(prepro_img)
            if is_cin(pred):
                st.image(image)
            else:
                st.error('The uploaded image is not of a Tunisian ID card, please reupload another image!')
                st.stop()
        else:
            rej_imgs=[]
            for file in files:
                image = Image.open(file)
                prepro_img = np.expand_dims(np.array(image.resize((224,224))),axis=0)
                pred = mobnet_model.predict(prepro_img)
                if is_cin(pred):
                    continue
                else:
                    rej_imgs.append(file.name)
            if rej_imgs:
                st.error(f'The following uploaded imgs are not of a Tunisian ID ({rej_imgs}), please re-upload others or remove them!')
                st.stop()
        tf.keras.backend.clear_session()
        del mobnet_model   


    if files != None:
        process = st.button('Process')
        if process:
            ocr_results = {'cin_number':[],'last_name':[],'first_name':[],'date_of_birth':[],'place_of_birth':[],'mother_name':[],'job':[],'address':[],'cin_date':[]}
            ocr_results_df = pd.DataFrame.from_dict(ocr_results)
            conf_score_list=[]
            if imgs_num > 1:
                st.write('Processing images...')
                progress_bar_files = st.progress(0)
            for f, file in enumerate(files):
                # Start of process
                ## Running YOLOv5 to dectect text fields
                image = Image.open(file)
                if imgs_num  == 1:
                    with st.spinner(text="Detecting text fields..."):
                        progress_bar = st.progress(0)
                        image_file = np.array(image)
                        model = torch.hub.load('ultralytics/yolov5', 'custom', path='./best.pt')
                        model.conf = 0.85
                        results = model(image_file)
                        results_df = results.pandas().xyxy[0]
                        progress_bar.progress(100)
                    st.success('Done!')
                else:
                    image_file = np.array(image)
                    model = torch.hub.load('ultralytics/yolov5', 'custom', path='./best.pt')
                    model.conf = 0.85
                    results = model(image_file)
                    results_df = results.pandas().xyxy[0]

                if imgs_num == 1:
                    ## Plotting text detection results
                    fig, ax = plt.subplots()
                    ax.imshow(image_file)
                    for i in range(results_df.shape[0]):
                        xy = (results_df.iloc[i,0], results_df.iloc[i,1])
                        hight = results_df.iloc[i,3]-results_df.iloc[i,1]
                        width = results_df.iloc[i,2]-results_df.iloc[i,0]
                        box_color = 'g' if results_df.iloc[i,4]>0.9 else 'y'
                        rect = patches.Rectangle(xy, width, hight, linewidth=1.5, edgecolor=box_color, facecolor='none')
                        ax.text(results_df.iloc[i,0], results_df.iloc[i,1]-10, s=f'{results_df.iloc[i,4]:.2f}', fontsize=5).set_bbox(dict(facecolor='white', alpha=0.7, boxstyle='Round4', edgecolor='none'))
                        ax.text(results_df.iloc[i,0]+(results_df.iloc[i,2]-results_df.iloc[i,0])/2, results_df.iloc[i,1]-10, s=f'{results_df.iloc[i,6]}', fontsize=5).set_bbox(dict(facecolor='white', alpha=0.7, boxstyle='Square', edgecolor='none'))
                        ax.add_patch(rect)
                    st.pyplot(fig)

                ## Preprocessing detected text fields for OCR (cropping, grayscale, thresholding)
                preproc_imgs, label_list = preprocess(results)
            
                ## Performing OCR
                easyocr_results={}
                if imgs_num == 1:
                    with st.spinner(text="Performing OCR..."):
                        progress_bar_ocr = st.progress(0)
                        for i, img in enumerate(preproc_imgs):
                            temp_res = arabic_ocr(preproc_imgs[i])
                            temp_res = [temp_res, label_list[i][0]]
                            easyocr_results[f'Object {i}'] = temp_res
                            progress_bar_ocr.progress((i+1)/len(preproc_imgs))
                    st.success('Done!')
                else:
                    for i, img in enumerate(preproc_imgs):
                        temp_res = arabic_ocr(preproc_imgs[i])
                        temp_res = [temp_res, label_list[i][0]]
                        easyocr_results[f'Object {i}'] = temp_res
                ## Cleaning and sorting OCR results (cleaning, sorting, and concatenating extracted text of same object. Then unifying texts that are of the same label ie: unifying long names) 
                easyocr_results_sorted = copy.deepcopy(easyocr_results)
                easyocr_results_sorted = clean_sort_results(easyocr_results_sorted, label_list)

                ## Creating a dataframe of the obtained results
                ocr_results = {'cin_number':[],'last_name':[],'first_name':[],'date_of_birth':[],'place_of_birth':[],'mother_name':[],'job':[],'address':[],'cin_date':[]}
                Confidence_score=0
                i = 0
                
                for  key in ocr_results.keys():
                    filled = False
                    for obj in easyocr_results_sorted.values():
                        if key == obj[3]:
                            ocr_results[key].append(obj[0])
                            Confidence_score += np.round(obj[1],4)
                            i+=1
                            filled = True
                    if filled == False:
                        ocr_results[key].append('Missing')
                Confidence_score = Confidence_score/i
                temp = pd.DataFrame.from_dict(ocr_results)
                ocr_results_df = ocr_results_df.append(temp, ignore_index=True)
                conf_score_list.append(np.round(Confidence_score,4))
                if imgs_num > 1:
                    progress_bar_files.progress((f+1)/len(files))
            
            if imgs_num > 1:
                st.write('Done!')
            ocr_results_df['conf_score'] = conf_score_list
            st.dataframe(ocr_results_df)
            #st.write(f'Confidence score: {np.round(Confidence_score,4)}')

            @st.cache
            def convert_df_csv(df):
                return df.to_csv().encode('utf-8')

            csv = convert_df_csv(ocr_results_df)

            st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='cin_data.csv',
            mime='text/csv')
    return None