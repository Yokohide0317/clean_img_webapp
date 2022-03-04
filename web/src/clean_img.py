import os
import sys
import cv2
from src.upcunet_v3 import RealWaifuUpScaler
from time import time as ttime

class Clean():
    """
    モデルを使ってアニメ調の画像をきれいにします。

    main(file_name, output_name, gpu=False)

    """

    def __init__(self):
        model_name = "up2x-latest-no-denoise.pth"
        path_name = os.path.dirname(__file__)
        self.model_path = os.path.join(path_name, "model", model_name)

    def main(self, file_name, output_name, gpu="False"):
        device = "cuda:0" if gpu else "cpu"
        upscaler = RealWaifuUpScaler(2, self.model_path, half=False, device=device)
        Tile = 4
        Amplification = 2

        output_name = ".".join(file_name.split(".")[:-1]) + "_cleaned." + file_name.split(".")[-1] if output_name == "False" else output_name 

        t0 = ttime()
        try:
            img = cv2.imread(file_name)[:, :, [2, 1, 0]]
            result = upscaler(img,tile_mode=2)
            #output_name = ".".join(file_name.split(".")[:-1]) + "_cleaned." + file_name.split(".")[-1]
            cv2.imwrite(output_name, result[:, :, ::-1])
        except RuntimeError as e:
            print ("Failed...")
            print (e)
        else:
            print("Done")
        t1 = ttime()
        print("Compleated", t1 - t0)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("-o", "--output", default="False")
    parser.add_argument("--gpu", action='store_true')
    args = parser.parse_args()

    print(f"Cleaning for: {args.file}")
    clean = Clean()
    clean.main(args.file, args.output, args.gpu)
