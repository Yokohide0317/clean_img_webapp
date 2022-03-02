import cv2
import datetime
import numpy as np
import os
from flask import Flask, render_template, request

from src import clean_img

app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
def hello_world():
    org_img_dir = "static/imgs/original/"
    cln_img_dir = "static/imgs/clean/"

    if request.method == 'GET':
        return render_template('index.html', img_path=None)

    elif request.method == 'POST':
        #### POSTにより受け取った画像を読み込む
        stream = request.files['img'].stream
        filename = str(request.files['img'].filename).split(".")[0]
        img_array = np.asarray(bytearray(stream.read()), dtype=np.uint8)
        img = cv2.imdecode(img_array, 1)
        #### 現在時刻を名前として「imgs/」に保存する
        dt_now = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
        #img_path = org_img_dir + dt_now + ".jpg"
        img_path = org_img_dir + filename + ".jpg"

        cv2.imwrite(img_path, img)
        out_path = cln_img_dir + filename + "_cleaned.jpg"

        clean = clean_img.Clean()
        clean.main(img_path, out_path, gpu=False)
        os.remove(img_path)
    #### 保存した画像ファイルのpathをHTMLに渡す
        return render_template('index.html', img_path=out_path) 

if __name__ == "__main__":
    app.run(debug=True)
