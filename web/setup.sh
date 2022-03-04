wget https://github.com/bilibili/ailab/releases/download/Real-CUGAN/updated_weights.zip
unzip updated_weights.zip
rm updated_weights.zip
mkdir -p src/model
mv updated_weights/up2x-latest-no-denoise.pth src/model
rm -r updated_weights
mkdir -p static/imgs/clean
mkdir -p static/imgs/original
