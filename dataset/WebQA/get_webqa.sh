#!/bin/bash
set -e

# Download WebQA image chunks (https://github.com/WebQnA/WebQA)
gdown --folder --remaining-ok https://drive.google.com/drive/folders/19ApkbD5w0I5sV1IeQ9EofJRyAjKnA7tb
gdown https://drive.google.com/uc?id=1SlYNpYYpwTfxIjQIDlM3o7a8kcpKO8PB -O WebQA_imgs_7z_chunks/

7z x WebQA_imgs_7z_chunks/imgs.7z.001
rm -rf WebQA_imgs_7z_chunks

# Extract images into .jpg files
python extract_image.py
rm imgs.tsv