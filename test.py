import requests
from PIL import Image
import numpy as np
import cv2
import base64
import json
import pickle

#test from GCR
resp = requests.post("https://ppe-detection-wx2bjo7cia-uc.a.run.app", files={'file': open('data/example-12.jpg', 'rb')})
#test locally on image built on external drive
# resp = requests.post("http://0.0.0.0:5000", files={'file': open('data/example-12.jpg', 'rb')})

print(resp.json())
# print(len(resp.json()))

# # Convert the pixels into an array using numpy
# array = np.array(resp.json(), dtype=np.uint8)
# print(array.shape)

# # Use PIL to create an image from the new array of pixels
# new_image = Image.fromarray(array)
# new_image.save('new.png')

imdata = base64.b64decode(resp.json()['image'])
print(imdata)
# im = pickle.loads(imdata)
# PILimage = Image.fromarray(im[...,::-1])
# PILimage.show()