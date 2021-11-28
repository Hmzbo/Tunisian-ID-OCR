import cv2
import numpy as np
import string
from PIL import Image

# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)
 
#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY +cv2.THRESH_OTSU )[1]

#dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
    
#erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

#invert edge detection
def invert(image):
    return 255-image

#skew correction
def deskew(img, limit, delta):
    white = (255,255,255)
    angles = np.arange(-limit, limit+delta, delta)
    scores = []
    for ang in angles:
        temp_RGBA = Image.fromarray(img).convert('RGBA')
        temp_img = temp_RGBA.rotate(ang, Image.NEAREST, expand = 1, fillcolor = white)
        temp_img = np.array(temp_img.convert('L'))
        hight,width = temp_img.shape
        a = int(hight*0.47)
        b = int(hight*0.53)
        upper=0
        lower =0
        for p in range(width):
            if temp_img[a,p]!=255:
                upper+=1
            if temp_img[b,p]!=255:
                lower+=1
        scores.append(upper+lower)
        
    best_angle = angles[scores.index(np.max(scores))]
    rotated_img = temp_RGBA.rotate(best_angle, Image.NEAREST, expand = 1, fillcolor = white)
    rotated_img = np.array(rotated_img.convert('L'))

    h,w = rotated_img.shape
    top_crop=0
    bot_crop=0
    for p in range(5,h//2):
        a=(h//2)-p
        b=(h//2)+p
        if (np.sum(rotated_img[a,:]==255)==w) & (top_crop==0):
            top_crop=a
        if (np.sum(rotated_img[b,:]==255)==w) & (bot_crop==0):
            bot_crop=b
    cropped_rotated_img = rotated_img[top_crop:bot_crop,:]            
    return cropped_rotated_img

def white_pad(image, pad_size):
    return cv2.copyMakeBorder(image.copy(),pad_size,pad_size,pad_size,pad_size,cv2.BORDER_CONSTANT,value=(255,255,255))

def cleanup_text(text):
  result = text.replace('\n\x0c','')
  result = result.replace('\u200f','')
  result = result.replace('\u200e','')
  result = result.translate(str.maketrans('','', string.punctuation))
  result = result.translate(str.maketrans('','', ''.join(list(string.ascii_lowercase))))
  result = result.translate(str.maketrans('','', ''.join(list(string.ascii_uppercase)))).strip()
  return result

# Utils for OCR

def preprocess (results):
    crops = results.crop(save=False)
    preproc_imgs = [crops[i]['im'] for i in range(len(crops))]
    label_list = [crops[i]['label'].split(' ') for i in range(len(crops))]
    for i, img in enumerate(preproc_imgs):
        pil_img = Image.fromarray(img)
        pil_img = pil_img.resize(np.multiply(pil_img.size,2))
        preproc_imgs[i] = thresholding(get_grayscale(np.array(pil_img)))
    return preproc_imgs, label_list

def clean_sort_results(easyocr_results, label_list):
    for obj_res in easyocr_results.values():
            for i in obj_res[0]:
                i[1] = cleanup_text(i[1])
    for l, obj_res in enumerate(easyocr_results.keys()):
        concat_text = ''
        conf_avg = []
        position_avg = []
        obj_list = easyocr_results[obj_res][0]
        for i in range(len(obj_list)):
            for j in range(i+1,len(obj_list)):
                if obj_list[j][0][2][0] > obj_list[i][0][2][0]:
                    aux = obj_list[i]
                    obj_list[i] = obj_list[j]
                    obj_list[j] = aux

        for i in easyocr_results[obj_res][0]:
            position_avg.append(np.array(i[0])) 
            concat_text = concat_text+' '+i[1]
            conf_avg.append(i[2])
        easyocr_results[obj_res] = [concat_text, np.mean(conf_avg), sum(position_avg)/len(position_avg), label_list[l][0]]

    pop_list = []
    object_list = list(easyocr_results.keys())
    for i in range(len(object_list)):
        for j in range(i+1,len(object_list)):
            if easyocr_results[object_list[i]][3] == easyocr_results[object_list[j]][3]:
                if easyocr_results[object_list[i]][2][0][1] < easyocr_results[object_list[j]][2][0][1]:
                    easyocr_results[object_list[i]][0] = easyocr_results[object_list[i]][0] + easyocr_results[object_list[j]][0]
                    pop_list.append(object_list[j])
                else:
                    easyocr_results[object_list[j]][0] = easyocr_results[object_list[j]][0] + easyocr_results[object_list[i]][0]
                    pop_list.append(object_list[i])
    for item in set(pop_list):
        easyocr_results.pop(item)
    return easyocr_results




