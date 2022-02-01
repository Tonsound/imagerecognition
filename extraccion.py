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


image = cv2.imread("Data/page35.jpg")
image = image[..., ::-1]

model = lp.Detectron2LayoutModel('lp://TableBank/faster_rcnn_R_50_FPN_3x/config',
                                 extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.3],
                                 label_map={0: "Table"}) #0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure" /'lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config'  'lp://TableBank/faster_rcnn_R_101_FPN_3x/config'


layout = model.detect(image)

prueba = lp.draw_box(image, layout, box_width=5)

#prueba.save("pruebapag35.png")

lp.elements.Layout
table_blocks = lp.Layout([b for b in layout if b.type=='Table'])

h, w = image.shape[:2]

left_interval = lp.Interval(0, w/2*1.05, axis='x').put_on_canvas(image)

left_blocks = table_blocks.filter_by(left_interval, center=True)
left_blocks.sort(key = lambda b:b.coordinates[1]) #, inplace=True

right_blocks = [b for b in table_blocks if b not in left_blocks]
right_blocks.sort(key = lambda b:b.coordinates[1]) #, inplace=True

# And finally combine the two list and add the index
# according to the order
table_blocks = lp.Layout([b.set(id = idx) for idx, b in enumerate(left_blocks + right_blocks)])


prueba2 =lp.draw_box(image, text_blocks,
            box_width=3,
            show_element_id=True)

prueba2.save("pruebapag35.png")


ocr_agent = lp.TesseractAgent(languages='eng')