FROM supervisely/base-py-sdk:6.36.0

RUN mkdir /ritm_models

# download checkpoints
RUN wget -O /ritm_models/sbd_h18_itermask.pth https://github.com/supervisely-ecosystem/ritm-interactive-segmentation/releases/download/v0.1/sbd_h18_itermask.pth
RUN wget -O /ritm_models/coco_lvis_h18s_itermask.pth https://github.com/supervisely-ecosystem/ritm-interactive-segmentation/releases/download/v0.1/coco_lvis_h18s_itermask.pth
RUN wget -O /ritm_models/coco_lvis_h18_itermask.pth https://github.com/supervisely-ecosystem/ritm-interactive-segmentation/releases/download/v0.1/coco_lvis_h18_itermask.pth
RUN wget -O /ritm_models/coco_lvis_h32_itermask.pth https://github.com/supervisely-ecosystem/ritm-interactive-segmentation/releases/download/v0.1/coco_lvis_h32_itermask.pth

# For PyTorch 1.10.0
RUN pip3 install --no-cache-dir setuptools==59.5.0
RUN pip3 install --no-cache-dir tqdm==4.66.4
# ---------------------------------------------- #

# install pytorch
RUN pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install diskcache==5.2.1 easydict tensorboard
RUN pip install opencv-python-headless==4.5.5.62 opencv-python==4.5.5.62 Pillow matplotlib imgaug>=0.4
RUN pip install albumentations==0.5.2 graphviz tqdm pyyaml
RUN pip install future cffi ninja
COPY dev_requirements.txt dev_requirements.txt
RUN pip install -r dev_requirements.txt
RUN pip install protobuf==3.20.3

LABEL python_sdk_version="6.73.486"