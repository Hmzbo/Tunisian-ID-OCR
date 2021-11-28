import streamlit as st
from cin_ocr import show_cin_ocr_page
from info import show_info_page

page = st.sidebar.selectbox('Page', options=('Info','Tunisian ID OCR'))

if page == 'Tunisian ID OCR':
    show_cin_ocr_page()
else:
    show_info_page()