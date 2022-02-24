ARG IMAGE
FROM $IMAGE

RUN mkdir /ritm_models

# download checkpoints
RUN wget -O /ritm_models/sbd_h18_itermask.pth https://github.com/supervisely-ecosystem/ritm-interactive-segmentation/releases/download/v0.1/sbd_h18_itermask.pth
RUN wget -O /ritm_models/coco_lvis_h18s_itermask.pth https://github.com/supervisely-ecosystem/ritm-interactive-segmentation/releases/download/v0.1/coco_lvis_h18s_itermask.pth
RUN wget -O /ritm_models/coco_lvis_h18_itermask.pth https://github.com/supervisely-ecosystem/ritm-interactive-segmentation/releases/download/v0.1/coco_lvis_h18_itermask.pth
RUN wget -O /ritm_models/coco_lvis_h32_itermask.pth https://github.com/supervisely-ecosystem/ritm-interactive-segmentation/releases/download/v0.1/coco_lvis_h32_itermask.pth

# install pytorch
RUN pip3 install torch==1.10.0+cu111 torchvision==0.11.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install supervisely==6.5.0 diskcache==5.2.1 easydict tensorboard
RUN pip3 install opencv-python-headless==4.5.5.62 opencv-python==4.5.5.62 Pillow matplotlib imgaug>=0.4
RUN pip3 install albumentations==0.5.2 graphviz tqdm pyyaml
RUN pip3 install future cffi ninja


RUN apt-get update && apt-get install -y openssh-server
EXPOSE 22

RUN apt-get install -y sudo
RUN mkdir -p /run/sshd

ARG home=/root
RUN mkdir $home/.ssh
COPY my_key.pub $home/.ssh/authorized_keys
RUN chown root:root $home/.ssh/authorized_keys && \
    chmod 600 $home/.ssh/authorized_keys

COPY sshd_daemon.sh /sshd_daemon.sh
RUN chmod 755 /sshd_daemon.sh
CMD ["/sshd_daemon.sh"]
ENTRYPOINT ["sh", "-c", "/sshd_daemon.sh"]