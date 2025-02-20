## **Description**

AIMET is a library designed to help developers optimize and deploy machine learning models efficiently. It provides tools for model compression, quantization, and other techniques to improve the performance and efficiency of AI models.

Here I have taken a pretrained resnet50 model and evaluated it with Image-net dataset  and then quantized the model using AIMET (AI Model Efficiency Toolkit) and then evaluated and compared the baseline (FP32) and quantized (W8A8)  model’s Top1 and Top5 accuracy

## **Model Used**

- ResNet 50

Model Path : `"/media/bmw/shabari/aimet/artifacts/resnet50.pth”`

## **Dataset Used**

- ImageNet

Dataset Path : `"/media/bmw/datasets/imagenet-1k/val”`

## **Usage**

Here is an example of how to use AIMET:

- Setup the configurations in config.json file

```bash
python src/main.py
```

## Result

| Models | Top 1 | Top 5 |
| --- | --- | --- |
| Baseline Resnet50 | 78.88 % | 94.80 % |
| W8A8 Quantized Resnet50 | 74.42 % | 92.34 % |