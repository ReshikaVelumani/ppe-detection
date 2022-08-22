import requests

#test from GCR
resp = requests.post("https://ppe-detection-wx2bjo7cia-uc.a.run.app", files={'file': open('data/example-12.jpg', 'rb')})
#test locally on image built on external drive
#resp = requests.post("http://0.0.0.0:5000", files={'file': open('../../../Documents/junkboatpic.jpg', 'rb')})

print(resp.json())
