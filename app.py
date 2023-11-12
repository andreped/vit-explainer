import requests
import re

import gradio as gr
import numpy as np
from torch import topk
from torch.nn.functional import softmax
from transformers import ViTImageProcessor, ViTForImageClassification
from transformers_interpret import ImageClassificationExplainer


def load_label_data():
    file_url = "https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt"
    response = requests.get(file_url)
    labels = []
    pattern = '["\'](.*?)["\']'
    for line in response.text.split('\n'):
        try:
            tmp = re.findall(pattern, line)[0]
            labels.append(tmp)
        except IndexError:
            pass
    return labels


class WebUI:
    def __init__(self):
        super().__init__()
        self.nb_classes = 10
        self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        self.labels = load_label_data()
    
    def run_model(self, image):
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        outputs = softmax(outputs.logits, dim=1)
        outputs = topk(outputs, k=self.nb_classes)
        return outputs

    def classify_image(self, image):
        top10 = self.run_model(image)
        return {self.labels[top10[1][0][i]]: float(top10[0][0][i]) for i in range(self.nb_classes)}

    def explain_pred(self, image):
        image_classification_explainer = ImageClassificationExplainer(model=self.model, feature_extractor=self.processor)
        saliency = image_classification_explainer(image)
        saliency = np.squeeze(np.moveaxis(saliency, 1, 3))
        saliency[saliency >= 0.05] = 0.05
        saliency[saliency <= -0.05] = -0.05
        saliency /= np.amax(np.abs(saliency))
        return saliency
    
    def run(self):
        examples=[
            ['https://github.com/andreped/INF1600-ai-workshop/releases/download/Examples/cat.jpg'],
            ['https://github.com/andreped/INF1600-ai-workshop/releases/download/Examples/dog.jpeg'],
        ]
        with gr.Blocks() as demo:
            with gr.Row():
                image = gr.Image(height=512)
                label = gr.Label(num_top_classes=self.nb_classes)
                saliency = gr.Image(height=512, label="saliency map", show_label=True)

                with gr.Column(scale=0.2, min_width=150):
                    run_btn = gr.Button("Run analysis", variant="primary", elem_id="run-button")

                    run_btn.click(
                        fn=lambda x: self.explain_pred(x),
                        inputs=image,
                        outputs=saliency,
                    )

                    run_btn.click(
                        fn=lambda x: self.classify_image(x),
                        inputs=image,
                        outputs=label,
                    )

                    gr.Examples(
                        examples=[
                            ['https://github.com/andreped/INF1600-ai-workshop/releases/download/Examples/cat.jpg'],
                            ['https://github.com/andreped/INF1600-ai-workshop/releases/download/Examples/dog.jpeg'],
                        ],
                        inputs=image,
                        outputs=image,
                        fn=lambda x: x,
                        cache_examples=True,
                    )
        
        demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=False)


def main():
    ui = WebUI()
    ui.run()


if __name__ == "__main__":
    main()
