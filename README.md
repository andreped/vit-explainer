---
title: 'vit-explainer'
colorFrom: indigo
colorTo: indigo
sdk: gradio
app_port: 7860
emoji: 🔥
pinned: false
license: mit
app_file: app.py
---

# vit-explainer

[![license](https://img.shields.io/github/license/DAVFoundation/captain-n3m0.svg?style=flat-square)](https://github.com/raidionics/AeroPath/blob/main/LICENSE.md)
[![CI/CD](https://github.com/andreped/vit-explainer/actions/workflows/deploy.yml/badge.svg)](https://github.com/andreped/vit-explainer/actions/workflows/deploy.yml)
<a target="_blank" href="https://huggingface.co/spaces/andreped/vit-explainer"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-yellow.svg"></a>

This web app was developed to test and demonstrate an explainable AI (XAI)-based technique with a pretrained vision transformer (ViT). 

The app is deployed on Hugging Face and can be access by clicking the 🤗 badge above.

## Demo

<img width="1283" alt="Screenshot 2023-11-12 at 23 48 54" src="https://github.com/andreped/vit-explainer/assets/29090665/1fe47bb5-625d-4717-9348-53930d5129dc">

## Development

To run the app locally, you run the following in the terminal:

```
git clone "https://github.com/andreped/vit-explainer.git"
cd "vit-explainer/"

virtualenv -ppython3 venv --clear
source venv/bin/activate
pip install -r requirements.txt

python app.py
```

## License

The code in this repository is released under [MIT license](https://github.com/andreped/vit-explainer/blob/main/LICENSE.md).
