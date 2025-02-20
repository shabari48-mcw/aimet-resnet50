
import torch
from data_pipeline import DataPipeline
import torchvision.models as models


## Aimet Imports
from aimet_torch.model_preparer import prepare_model  
from aimet_torch.model_validator.model_validator import ModelValidator
from aimet_torch.quantsim import QuantizationSimModel
from aimet_common.defs import QuantScheme
from aimet_torch.batch_norm_fold import fold_all_batch_norms

from logger import logger

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Quantize():
    
    def __init__(self,model,activation_bits:int,weight_bits:int):
        self.model=model
        self.activation_bits=activation_bits
        self.weight_bits=weight_bits
    
    #Load Model
    def load_model(model_path:str):
        model = models.resnet50()
        model.load_state_dict(torch.load(model_path,weights_only=True))
        return model


    ### Prepare the Model according to Model Guidelines using Model Preparator API
    def prepare_model(model:torch.nn.Module)->torch.nn.Module:
        """
        Function to prepare the model
        Args:
            model : torch.nn.Module : Model to be prepared
        Returns:
            torch.nn.Module : Prepared Model
        """
        logger.info("Preparing Model")
        model = prepare_model(model)
        logger.info("Model Prepared")
        return model

    # ### Validate the Model using ModelValidator API

    def validate_model(model: torch.nn.Module,input_tensor: torch.Tensor):
        """
        Function to validate the model
        Args:
            model : torch.nn.Module : Model to be validated
            input_tensor : torch.Tensor : Input tensor to the model
        """
        logger.info("Validating Model")
        ModelValidator.validate_model(model.to(device), model_input=input_tensor.to(device))
        logger.info("Model Validated")
        
        
    @staticmethod
    ## Pass Calibration Data (Unlabelled Data)
    def pass_calibration_data(sim_model,data_params:dict):
        """
        Function to pass calibration
        Args:
        sim_model : QuantizationSimModel : Model to be quantized
        use_cuda : bool : Use Cuda
        """
        
        logger.info("Passing Calibration Data")
        logger.info("Data Params: {}".format(data_params))
        data_loader = DataPipeline.get_val_dataloader(data_params)
        batch_size = data_loader.batch_size

        sim_model.eval()
        samples = 1000

        batch_cntr = 0
        with torch.no_grad():
            for input_data, target_data in data_loader:

                inputs_batch = input_data.to(device)
                sim_model(inputs_batch)

                batch_cntr += 1
                if (batch_cntr * batch_size) > samples:
                    break
        logger.info("Calibration Data Passed")
                

    ## Quantize the Model using QuantizationSimModel API
    def quantize_model(model:torch.nn.Module,data_params: dict,input_tensor:torch.Tensor)->QuantizationSimModel:
        
        """
        Function to quantize the model
        Args:
            model : torch.nn.Module : Model to be quantized
            data_params : dict : Data Parameters
            input_tensor : torch.Tensor : Input Tensor
        Returns:
            QuantizationSimModel : Quantized Model
        """
        
        # Perform Batch Normalization Folding before quant simulation 
        # as it improves inference performance on quantized runtimes
        
        weight_bw = data_params['Quantization.weight_bits']
        activation_bw = data_params['Quantization.activation_bits']
        
        logger.info("Folding Batch Norms")
        _ = fold_all_batch_norms(model.to(device), input_shapes=(1, 3, 224, 224),dummy_input=input_tensor.to(device))
        logger.info("Batch Norms Folded")
        
        logger.info("Quantizing Model")
        sim=QuantizationSimModel(model.to(device), dummy_input=input_tensor.to(device),
                                quant_scheme=QuantScheme.training_range_learning_with_tf_init,  
                                default_output_bw=activation_bw,
                                default_param_bw=weight_bw)
        logger.info("Model Quantized")
        
        logger.info("Computing Encodings")
        sim.compute_encodings(forward_pass_callback=Quantize.pass_calibration_data,
                        forward_pass_callback_args=data_params)
        logger.info("Encodings Computed")
        
        return sim


