import streamlit as st

def show_info_page():
    st.markdown('''
    # Tunisian ID ORC Web Application

    This web application is a demo app made with streamlit to showcase the process of Optical Chracter Recognition on the Tunisian ID (CIN).
    To use this application you need to have one or many images of CINs in the indicated formats i.e: .png, .jpg, .jpeg

    # OCR process
    The uploaded image will go through a process that can be devided in three main parts:

    1. Image classification
    2. Text field detection
    3. Information extraction

    ## Image classification
    A classification model is used to predict whether the imported image(s) is of a Tunisian ID card. In case one or more 
    uploaded image(s) are not of a Tunisian ID the user is asked to whether remove the image(s) or re-upload other one(s).

    ## Text field detection
    Once the image(s) are accepted to be of Tunsian ID(s), an object detection model is then used to detect the text fields in
    the images. The detected fields are then cropped out of each image, preprocessed, and passed to the next step.

    ## Information Extraction
    This is the final step of the process, the preprocessed image(s) are fed to an OCR model to extract the texts and then
    few post-processing techniques are used sort and rearrange the output of the OCR model as well as to assign each one
    corresponding label.
    The final results are then printed on screen and available for download on a .csv format.
    ''')
    return None