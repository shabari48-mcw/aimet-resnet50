import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
from tqdm import tqdm
from logger import logger

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DataPipeline:
    
    def __init__(self,data_path:str,model_path:str
                 ,batch_size:int,num_classes:int,
                 images_per_class:int):
        
        self.batch_size=batch_size
        self.num_classes=num_classes
        self.images_per_class=images_per_class
        self.data_path=data_path
        self.model_path=model_path

    
    @staticmethod
    def get_val_dataloader(config: dict) -> DataLoader:
        
        logger.info(f"Config: {config}")
        data_path = config['Path.DATA_DIR']
        batch_size = config['Dataloader.batch_size']
        num_classes = config['Dataloader.num_of_classes']
        images_per_class = config['Dataloader.images_per_class']
        
        logger.info("Creating Validation Dataloader")
        logger.info(f"Data Path: {data_path} Batch Size: {batch_size} Num Classes: {num_classes} Images Per Class: {images_per_class}")
        logger.info("Transforming Data")
        
        tranform_data= transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])
        logger.info("Data Transformed")
        
        
        logger.info("Creating ImageFolder Dataset")
        dataset = ImageFolder(
            root=data_path,
            transform=tranform_data
        )
        logger.info("ImageFolder Dataset Created")
        
        logger.info("Creating Dataloader")
        
        all_classes=dataset.classes
        
        # Filter dataset to include only a subset of classes
        selected_classes = all_classes[:num_classes]
        
        # Create a mapping of selected class names to their indices
        selected_class_indices = [dataset.class_to_idx[class_name] for class_name in selected_classes]
        
        # Get indices of samples belonging to selected classes
        selected_indices = []
        class_counts = {idx: 0 for idx in selected_class_indices}
        
        for idx, (_, label) in enumerate(dataset.samples):
            if label in selected_class_indices and class_counts[label] < images_per_class:
                selected_indices.append(idx)
                class_counts[label] += 1
                
        # Create a subset dataset
        subset_dataset = Subset(dataset, selected_indices)
        
        
        dataloader = DataLoader(
            subset_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )
        
        logger.info("Dataloader Created")
        
        return dataloader
        

    
    @staticmethod
    def evaluate(model : torch.nn.Module,data_params:dict) -> dict :
      
        top1_correct = 0
        top5_correct = 0
        total_samples = 0
        
        logger.info("Evaluating Model")


        with torch.no_grad():
            
            logger.info("Starting Evaluation")
            for image, label in tqdm(DataPipeline.get_val_dataloader(data_params), desc="Evaluating"):
    
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
        
        
        # Calculate final metrics
        top1_accuracy = (top1_correct / total_samples) * 100
        top5_accuracy = (top5_correct / total_samples) * 100

        print(f"Top-1 Accuracy: {top1_accuracy:.2f}%")  
        print(f"Top-5 Accuracy: {top5_accuracy:.2f}%")
        print(f"Total Samples Evaluated: {total_samples}")
        