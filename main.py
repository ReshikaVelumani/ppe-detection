import video_demo
from flask import Flask
from flask_cors import CORS
from flask import request, jsonify
import io
from PIL import Image
import numpy as np


# instantiate the app
app = Flask(__name__)
cors = CORS(app)

@app.route('/', methods=['GET', 'POST'])
def WarningMsg():
    if request.method == "POST":
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({"error": "no file"})

        try:
            file_name = file.read()
            image = np.array(Image.open(io.BytesIO(file_name))) 
            warning = video_demo.get_warning(image)
            return warning
        except Exception as e:
            return jsonify({"error": str(e)})
    else:
        return 'OK' 


if __name__ == '__main__':
    app.run(debug=True, port=5000)