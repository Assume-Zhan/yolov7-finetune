# Required Libraries
from utils.datasets import create_dataloader
from utils.general import check_img_size

# Some check definitions
CHECK_DATALOADER = True

# Setup hyper-parameters
class Settings():
    def __init__(self, setting_dict):
        self.path = setting_dict['path']
        self.image_size = setting_dict['image_size']
        self.single_cls = False
        self.batch_size = setting_dict['batch_size']

# Get dataset
def get_dataloader(settings, stride):
    
    # Return dataset and dataloader
    return create_dataloader(
        settings.path, 
        settings.image_size, 
        settings.batch_size, 
        stride, 
        settings
    )

# train function
def train(opt):
    
    # Load data
    stride = 32
    settings = {
        'path': "/yolov7/data/Aquarium/train/images",
        'image_size': [768, 1024],
        'batch_size': 1
    }
    settings['image_size'], imgsz_test = [check_img_size(x, stride) for x in settings['image_size']]
    settings = Settings(settings)
    
    dataloader, dataset = get_dataloader(settings, stride)
    
    # Prepare Model
    
    # Prepare Optimizier
    
    # Prepare LR scheduler
    
    # Start training
    
    # Test
    

# Main
if __name__ == "__main__":
    if CHECK_DATALOADER == True:
        stride = 32
        settings = {
            'path': "/yolov7/data/Aquarium/train/images",
            'image_size': [768, 1024],
            'batch_size': 1
        }
        settings['image_size'], imgsz_test = [check_img_size(x, stride) for x in settings['image_size']]
        
        settings = Settings(settings)
        
        dataloader, dataset = get_dataloader(settings, stride)
    else:
        
       train(opt)