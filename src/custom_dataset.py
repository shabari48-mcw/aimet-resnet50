import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from logger import logger
from torch.utils.data import Subset

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class CustomLoader():
    
        def __init__(self,params:dict):
            self.data=params['DATA_DIR']
            self.batch_size=params['batch_size']
            self.num_classes=params['num_classes']
            self.images_per_class=params['images_per_class']
            self.params=params
            
            
        def get_dataloader(self) -> DataLoader:
            
            tranform_data= transforms.Compose([
                    transforms.Resize((224,224)),
                    transforms.ToTensor()
                ])
            
    
            logger.info("Creating Dataset")
            dataset = ImageFolder(
                root=self.data,
                transform=tranform_data
            )
            
            logger.info(" Dataset Created")
            
            logger.info("Creating Dataloader")
            
            all_classes=dataset.classes
            
            # Filter dataset to include only a subset of classes
            selected_classes = all_classes[:self.num_classes]
            
            # Create a mapping of selected class names to their indices
            selected_class_indices = [dataset.class_to_idx[class_name] for class_name in selected_classes]
            
            # Get indices of samples belonging to selected classes
            selected_indices = []
            class_counts = {idx: 0 for idx in selected_class_indices}
            
            for idx, (_, label) in enumerate(dataset.samples):
                if label in selected_class_indices and class_counts[label] < self.images_per_class:
                    selected_indices.append(idx)
                    class_counts[label] += 1
                    
            # Create a subset dataset
            subset_dataset = Subset(dataset, selected_indices)
            
            dataloader = DataLoader(
                subset_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=4
            )
            
            logger.info("Dataloader Created")
            
            return dataloader