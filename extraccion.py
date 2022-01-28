import layoutparser as lp
import cv2

from pdf2image import pdfinfo_from_path, convert_from_path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



images = convert_from_path('Data/MC0018882-11-20.pdf', poppler_path = r'C:\Program Files\poppler-0.68.0\bin')

for i in range(len(images)):
      # Save pages as images in the pdf
    images[i].save('Data/page'+ str(i) +'.jpg', 'JPEG')


image = cv2.imread("Data/page0.jpg")
image = image[..., ::-1]
