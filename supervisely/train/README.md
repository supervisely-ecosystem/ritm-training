<div align="center" markdown> 

<img src="https://user-images.githubusercontent.com/106374579/187906520-671c4261-8253-449b-94cd-03227968b077.png"/>

# Train RITM
  
<p align="center">

  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#Related-Apps">How To Run</a> •
  <a href="#Video">Video</a> •
  <a href="#Screenshot">Screenshot</a> •
  <a href="#Acknowledgment">Acknowledgment</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](../../../../supervisely-ecosystem/ritm-training/supervisely/train)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervisely.com/slack) 
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/ritm-training)
[![views](https://app.supervisely.com/img/badges/views/supervisely-ecosystem/ritm-training/supervisely/train.png)](https://supervisely.com)
[![runs](https://app.supervisely.com/img/badges/runs/supervisely-ecosystem/ritm-training/supervisely/train.png)](https://supervisely.com)

</div>

## Overview 

🔥🔥🔥 Check out our [youtube tutorial](https://youtu.be/7oEf_R74-z0) and the [complete guide in our blog](https://supervisely.com/blog/custom-smarttool-wheat/):   

<a href="https://youtu.be/Rsr8xWJ6s9I" target="_blank"><img src="https://user-images.githubusercontent.com/106374579/258399115-1eac5ad8-d292-470b-8e8b-e468f26f7adb.png"/></a>

This app provides dashboard for training RITM models in Supervisely. 

Available 4 pretrained checkpoints based on HRNet for interactive click-based segmentation. This app helps you fine-tune these checkpoints on your data and speed up labeling process at times.


## How To Run

1. Add app to your team from Ecosystem
2. Be sure that you connected computer with GPU to your team by running Supervisely Agent on it
3. Run app from context menu of images project
4. Push "Finish training" button when training will be completed if you don't want to continue training.

<img src="https://i.imgur.com/q9fHzV7.png" />


## Related Apps

You can use served model in next Supervisely Applications ⬇️ 
  

- [RITM interactive segmentation SmartTool](../../../../supervisely-ecosystem/supervisely-ecosystem%2Fritm-interactive-segmentation%2Fsupervisely) - app allows to apply your trained model to your data in real time interactively. Just run app and specify path to your custom model, open your data project and enjoy fast labeling!
   
    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/ritm-interactive-segmentation/supervisely" src="https://i.imgur.com/CCnlZJP.png" width="350px"/> 

- [Flying Objects](../../../../supervisely-ecosystem/flying-objects) - app allows to generate synthetic data for training RITM model. Configure background images and labeled segments and you'll get a mix. Flexible app setting allow train strong models even on an extremely small labeled data.
   
    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/flying-objects" src="https://i.imgur.com/x5cafOU.png" width="350px"/>

## Video

In progress. Will be available soon.

## Screenshot

<img src="https://i.imgur.com/oZb4C6G.png"/>

## Acknowledgment

This app is based on the great work `RITM` ([github](https://github.com/saic-vul/ritm_interactive_segmentation)). ![GitHub Org's stars](https://img.shields.io/github/stars/saic-vul/ritm_interactive_segmentation?style=social)
