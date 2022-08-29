import requests
from PIL import Image
import numpy as np
import cv2
import base64
import json
import pickle

#test from GCR
# resp = requests.post("https://ppe-detection-wx2bjo7cia-uc.a.run.app", files={'file': open('data/example-12.jpg', 'rb')})
#test locally on image built on external drive
resp = requests.post("http://0.0.0.0:5000", files={'file': open('data/example-2.jpg', 'rb')})

# print(resp.json())
print('OK')