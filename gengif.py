from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import imageio
from PIL import Image
import matplotlib.image as mpimg
import glob
import os

result = []
paths = glob.glob("./iNeRFrender21_800")
os.makedirs('./pngs', exist_ok=True)
for idx , path in enumerate(paths) :
    img = cv2.imread(path)
    img = cv2.resize(img , (400, 1200) , interpolation = cv2.INTER_AREA)
    result.append(img)
    name = path.split(".jpg")[0]
    cv2.imwrite(f'./pngs/{name}.png' , img)    

path = [f"./pngs/{i}" for i in os.listdir("./pngs")]
paths = [ Image.open(i) for i in path]
imageio.mimsave('./test.gif', paths, fps=10)
