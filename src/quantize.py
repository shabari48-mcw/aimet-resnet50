
import torch
from tqdm import tqdm
import os
from logger import logger
from torch.utils.data import DataLoader
from custom_dataset import CustomLoader



## Aimet Imports
from aimet_torch.model_preparer import prepare_model  
from aimet_torch.model_validator.model_validator import ModelValidator
from aimet_torch.quantsim import QuantizationSimModel
from aimet_common.defs import QuantScheme
from aimet_torch.batch_norm_fold import fold_all_batch_norms


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Quantize():
    
    def __init__(self,model:torch.nn.Module,quant_params:dict):
      
        self.model=model.to(device)
        c,h,w=quant_params['input_shape[0]'],quant_params['input_shape[1]'],quant_params['input_shape[2]']
        self.input_tensor=torch.randn(1,c,h,w,device=device,dtype=torch.float32)
        self.quant_params=quant_params
        self.quantized_model=None
    

    ### Prepare the Model according to Model Guidelines using Model Preparator API
    def prepare_model(self)->None:
        """
        Function to prepare the model
        Args:
            model : torch.nn.Module : Model to be prepared
        Returns:
            torch.nn.Module : Prepared Model
        """
        logger.info("Preparing Model")
        self.model = prepare_model(self.model)
        logger.info("Model Prepared")
    

    # ### Validate the Model using ModelValidator API

    def validate_model(self)->None:
        """
        Function to validate the model
        Args:
            model : torch.nn.Module : Model to be validated
        """
        logger.info("Validating Model")
        ModelValidator.validate_model(self.model.to(device), model_input=self.input_tensor)
        logger.info("Model Validated")
        
    @staticmethod
    ## Pass Calibration Data (Unlabelled Data)
    def pass_calibration_data(sim_model,data_loader:CustomLoader)->None:
        """
        Function to pass calibration
        Args:
        sim_model : QuantizationSimModel : Model to be quantized
        use_cuda : bool : Use Cuda
        """
        
        logger.info("Passing Calibration Data")
        
        logger.info("Starting Calibration")
        
        
        
        calibration_size = data_loader.params['calibration_size']
        logger.info("Calibration Size: ",calibration_size)
        batch_cntr = 0
        
        batch_size = data_loader.batch_size
        sim_model.eval()
        with torch.no_grad():
            for input_data, target_data in tqdm(data_loader.get_dataloader(),desc="Calibrating"):
                inputs_batch = input_data.to(device)
                sim_model(inputs_batch)

                batch_cntr += 1
                if (batch_cntr * batch_size) > calibration_size:
                    break
             
        logger.info("Calibration Data Passed")
                

    ## Quantize the Model using QuantizationSimModel API
    def quantize_model(self,calibration_dataloader:DataLoader
                       )->QuantizationSimModel:
        
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
        
        weight_bw = self.quant_params['weight_bits']
        activation_bw = self.quant_params['activation_bits']
        
        logger.info("Folding Batch Norms")
        _ = fold_all_batch_norms(self.model.to(device), input_shapes=self.input_tensor.shape,dummy_input=self.input_tensor)
        logger.info("Batch Norms Folded")
        
        logger.info("Quantizing Model")
        
        sim=QuantizationSimModel(self.model.to(device), dummy_input=self.input_tensor,
                                quant_scheme=QuantScheme.training_range_learning_with_tf_init,  
                                default_output_bw=activation_bw,
                                default_param_bw=weight_bw)
        logger.info("Model Quantized")
        
        logger.info("Computing Encodings")
        
        sim.compute_encodings(forward_pass_callback=Quantize.pass_calibration_data,
                              
                        forward_pass_callback_args=calibration_dataloader)
        
        logger.info("Encodings Computed")
        
        self.quantized_model=sim.model
        
        
        output_path='./artifacts'
        dummy_input = torch.randn(1, 3, 224, 224,dtype=torch.float32)
        dummy_input = dummy_input.cpu()
        sim.export(path=output_path, filename_prefix=f'resnet50_W{weight_bw}A{activation_bw}', dummy_input=dummy_input)
        return sim
    
    
    def get_quantized_model(self)->torch.nn.Module:
        return self.quantized_model
    

    @staticmethod
    def evaluate_model(model,dataloader: DataLoader,model_type:str) -> dict :
        """
        Function to evaluate the model
        Args:
            model : torch.nn.Module : Model to be evaluated
            data_params : dict : Data Parameters
        Returns:
            dict : Evaluation Metrics
        """
        
        top1_correct = 0
        top5_correct = 0
        total_samples = 0
        
        logger.info(f"Evaluating Model {model_type}")

        model.eval()

        with torch.no_grad():
            
            logger.info("Starting Evaluation")
            
            for image, label in tqdm(dataloader.get_dataloader(), desc="Evaluating"):

                outputs = model(image.to(device))
                
             
                # Calculate Top-1 and Top-5 accuracy
                _, pred = outputs.topk(5, 1, True, True)
                pred = pred.t()
                label = label.to(device)
                label = label.view(1, -1).expand_as(pred)
                
                correct = pred.eq(label)
             
                top1_correct += correct[0].sum().item()
                top5_correct += correct[:5].any(0).sum().item()
                total_samples += label.size(1)

        logger.info("Evaluation Complete")
        

        top1_accuracy = (top1_correct / total_samples) * 100
        top5_accuracy = (top5_correct / total_samples) * 100

        print(f"Top-1 Accuracy: {top1_accuracy:.2f}%")  
        print(f"Top-5 Accuracy: {top5_accuracy:.2f}%")
        print(f"Total Samples Evaluated: {total_samples}")
        
        
        logger.info(f"Model Type :{model_type} Top-1 Accuracy: {top1_accuracy:.2f}%  Top-5 Accuracy: {top5_accuracy:.2f}% Total Samples Evaluated: {total_samples}")
        
        with open("./log/evaluation_results.txt", "a") as f:
            f.write(f"---------Model Type: {model_type}-----------\n")
            f.write(f"Top-1 Accuracy: {top1_accuracy:.2f}%\n")
            f.write(f"Top-5 Accuracy: {top5_accuracy:.2f}%\n")
            f.write(f"Total Samples Evaluated: {total_samples}\n")
            
        
