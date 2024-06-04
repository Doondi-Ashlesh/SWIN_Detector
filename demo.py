import gradio as gr
from transformers import pipeline, AutoModelForImageClassification, AutoImageProcessor
import torch


def predict_fn(inp):
    image_processor = AutoImageProcessor.from_pretrained('./models/swin-tiny-complete')
    inputs = image_processor(inp, return_tensors='pt')

    model = AutoModelForImageClassification.from_pretrained('./models/swin-tiny-complete')
    with torch.no_grad():
        logits = torch.nn.functional.softmax(model(**inputs).logits, dim=1)

    print(logits)
    labels = model.config.id2label
    out = {}

    for i in range(5):
        out[labels[i]] = float(logits[0][i])

    return out


iface = gr.Interface(
    fn=predict_fn,
    inputs=gr.Image(type='pil'),
    outputs=gr.Label(num_top_classes=5),
    allow_flagging='never',
)

iface.launch()
