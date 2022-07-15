<div align="center" markdown> 

<img src="https://i.imgur.com/3U7GZmf.png"/>

# Train RITM
  
<p align="center">

  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#Related-Apps">How To Run</a> •
  <a href="#Video">Video</a> •
  <a href="#Screenshot">Screenshot</a> •
  <a href="#Acknowledgment">Acknowledgment</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/ritm-training/supervisely/train)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack) 
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/ritm-training)
[![views](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/ritm-training/supervisely/train&counter=views&label=views&123)](https://supervise.ly)
[![runs](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/ritm-training/supervisely/train&counter=runs&label=runs&123)](https://supervise.ly)

</div>

## Overview 

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
  

- [RITM interactive segmentation SmartTool](https://ecosystem.supervise.ly/apps/supervisely-ecosystem%2Fritm-interactive-segmentation%2Fsupervisely) - app allows to apply your trained model to your data in real time interactively. Just run app and specify path to your custom model, open your data project and enjoy fast labeling!
   
    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/ritm-interactive-segmentation/supervisely" src="https://i.imgur.com/CCnlZJP.png" width="350px"/> 

- [Flying Objects](https://ecosystem.supervise.ly/apps/flying-objects) - app allows to generate synthetic data for training RITM model. Configure background images and labeled segments and you'll get a mix. Flexible app setting allow train strong models even on an extremely small labeled data.
   
    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/flying-objects" src="https://i.imgur.com/x5cafOU.png" width="350px"/>

## Video

In progress. Will be available soon.

## Screenshot

<img src="https://i.imgur.com/oZb4C6G.png"/>

## Acknowledgment

This app is based on the great work `RITM` ([github](https://github.com/saic-vul/ritm_interactive_segmentation)). ![GitHub Org's stars](https://img.shields.io/github/stars/saic-vul/ritm_interactive_segmentation?style=social)