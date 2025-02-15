import os
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms
current_dir = os.path.dirname(os.path.abspath(__file__))
from abc import abstractmethod

class SegmentationModel(torch.nn.Module):
    def __init__(self, freeze=True, **build_kwargs):
        super().__init__()
        self.model, self.eval_transforms = self._build(**build_kwargs)
        self.confidence_thresh = 0.5

        # Set all parameters to be non-trainable
        if freeze and self.model is not None:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
            
    def forward(self, batch):
        '''
        Can be overwritten if model requires special forward pass.
        '''
        z = self.model(batch)
        return z
        
    @abstractmethod
    def _build(self, **build_kwargs):
        pass


class HESTSegmenter(SegmentationModel):

    def _build(self, checkpoint_dir, device):

        model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50')
        model.classifier[4] = nn.Conv2d(
            in_channels=256,
            out_channels=2,
            kernel_size=1,
            stride=1
        )

        # download is not prev. downloaded. 
        if not os.path.isfile(os.path.join(checkpoint_dir, 'deeplabv3_seg_v4.ckpt')):
            try:
                from huggingface_hub import snapshot_download
            except:
                raise Exception("Please install huggingface_hub (`pip install huggingface_hub`) to use this model")
            snapshot_download(repo_id="MahmoodLab/hest-tissue-seg", repo_type='model', local_dir=checkpoint_dir, cache_dir=checkpoint_dir)

        checkpoint = torch.load(os.path.join(checkpoint_dir, 'deeplabv3_seg_v4.ckpt'), map_location=torch.device('cpu'), weights_only=False)
            
        clean_state_dict = {}
        for key in checkpoint['state_dict']:
            if 'aux' in key:
                continue
            new_key = key.replace('model.', '')
            clean_state_dict[new_key] = checkpoint['state_dict'][key]
        model.load_state_dict(clean_state_dict)
        model.to(device)
        self.device = device
        self.input_size = 512

        eval_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        return model, eval_transforms
    
    def forward(self, image):
        # input should be of shape (batch_size, C, H, W)
        assert len(image.shape) == 4, f"Input must be 4D image tensor (shape: batch_size, C, H, W), got {image.shape} instead"
        logits = self.model(image.to(self.device))['out']
        softmax_output = F.softmax(logits, dim=1)
        predictions = (softmax_output[:, 1, :, :] > self.confidence_thresh).to(torch.uint8)  # Shape: [bs, 512, 512]
        return predictions
        
 

def segmentation_model_factory(model_name, device, freeze=True):
    '''
    Build a slide encoder based on model name.
    '''
    if model_name == 'hest':
        return HESTSegmenter(freeze, checkpoint_dir = os.path.join(current_dir, 'hest-tissue-seg/'), device = device)
    else:
        raise ValueError(f"Model type {model_name} not supported")
