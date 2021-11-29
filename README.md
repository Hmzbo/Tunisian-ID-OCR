# Tunisian ID OCR
Tunisian ID OCR is a web application made with python to perform OCR on Tunisian ID cards. For more information about this project please refer to the Info page on the web application, or the ```Info.pdf``` file.
The application is deployed on the Streamlit Cloud, and can be accessed via this [link](https://share.streamlit.io/hmzbo/tunisian-id-ocr/main/app.py).
## How to use
### Locally
To use the application, you can either clone this repository to your local machine and run it using the command ```streamlit run app.py```, however all the required libraries sit in the ```requirement.txt``` file should be installed beforehand.
### On the cloud
The web application is directly accessible via this [link](https://share.streamlit.io/hmzbo/tunisian-id-ocr/main/app.py), however it takes time to load if it hasn't been accessed for a while.

Once the application is running, you can read the info page to have a better understanding of how it works, then you can perform the OCR process on the Tunisian ID OCR page which is accessible via the sidebar menu. 
To visualize the steps taken by the application during the OCR process you need to upload only one image using the upload button. However you can always perform this process on a batch of images at the same time.

The accuracy of the OCR is 81% on average, however the process takes quite a long time (~7 sec per image).
