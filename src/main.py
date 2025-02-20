from logger import logger
from parse_json import parse_json
import os
import torch
from quantize import Quantize
import argparse
from custom_dataset import CustomLoader



from torchvision.models import resnet50


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description='Quantization of Pytorch Models')
    parser.add_argument('-c','--config', type=str, help='Path to the config file', default='/media/bmw/shabari/aimet/config/config.json')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    json_path = os.path.join(args.config)
    
    params= parse_json(json_path)
    
    logger.info(f"Parameters: {params}")
    
    model=resnet50()
    model.load_state_dict(torch.load(params['MODEL_DIR']))
    
    
    dataloader= CustomLoader(params)
    
    
    model=Quantize(model,params)
    
    model.prepare_model()
    
    model.validate_model()
    
    logger.info("Model Evaluation for Base Line Model")
    
    model.evaluate_model(model.model, dataloader,"Base Line Model")
    
    model.quantize_model(dataloader)

    quant_model=model.get_quantized_model()
    
    model.evaluate_model(quant_model, dataloader,"Quantized Model")

    
    
if __name__ == '__main__':
    main()
 