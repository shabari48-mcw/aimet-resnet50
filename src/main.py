from logger import logger
from parse_json import parse_json
import os
import torch
from pathlib import Path
from data_pipeline import DataPipeline
from quantize import Quantize


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    json_path = os.path.join(Path("/media/bmw/shabari/aimet/config/config.json"))
    parsed_data = parse_json(json_path)
    
    logger.info(f"Parsed Data : {parsed_data}")
    
    data_pipeline = DataPipeline(parsed_data['Path.DATA_DIR'],parsed_data['Path.MODEL_DIR'],
                                 parsed_data['Dataloader.batch_size'],
                                 parsed_data['Dataloader.num_of_classes'],
                                 parsed_data['Dataloader.images_per_class'])
    
    model=Quantize.load_model(data_pipeline.model_path)
    
    model.eval()
    
    DataPipeline.evaluate(model.to(device),data_params=parsed_data)
    
    quant=Quantize(model,parsed_data['Quantization.activation_bits'],parsed_data['Quantization.weight_bits'])
    input_tensor=torch.randn(1,3,224,224)
    model=Quantize.prepare_model(model)
    
    Quantize.validate_model(model,input_tensor)
    
    
    logger.warn(f"Parsed Data: {parsed_data}")
    sim_model = Quantize.quantize_model(model,parsed_data,input_tensor)
    
    DataPipeline.evaluate(sim_model.model,data_params=parsed_data)
    
    
if __name__ == '__main__':
    main()
 