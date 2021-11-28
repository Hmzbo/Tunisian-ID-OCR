from easyocr import Reader
import string


def arabic_ocr(image):
  # break the input languages into a comma separated list
  langs = "ar,en".split(",")
  #print("[INFO] OCR'ing with the following languages: {}".format(langs))
# OCR the input image using EasyOCR
  print("[INFO] OCR'ing input image...")
  reader = Reader(langs)
  results = reader.readtext(image)
  print("Results:")
  print(results)
  return results
