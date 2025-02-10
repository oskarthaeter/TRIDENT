import sys
import os
import torch
import traceback
from abc import abstractmethod
from einops import rearrange
from trident.IO import get_weights_path
import warnings

"""
This file contains an assortment of pretrained slide encoders, including THREADS (ours) and baselines.
"""

def encoder_factory(model_name, pretrained=True, freeze=True, **kwargs):
        '''
        Build a slide encoder model.

        Args:
            model_name (str): Name of the model to build.
            pretrained (bool): Whether to load pretrained weights.
            freeze (bool): Whether to freeze the weights of the model.
            **kwargs: Additional arguments to pass to the model constructor.

        Returns:
            torch.nn.Module: The slide encoder model.
        '''

        if model_name.startswith('mean-'):
            enc = MeanSlideEncoder
        elif 'threads' in model_name:
            # raise ValueError(f"threads is not public. Coming soon!")
            enc = ThreadsSlideEncoder
        elif 'titan' in model_name:
            enc = TitanSlideEncoder
        elif 'prism' in model_name:
            enc = PRISMSlideEncoder
        elif 'chief' in model_name:
            enc = CHIEFSlideEncoder
        elif 'gigapath' in model_name:
            enc = GigaPathSlideEncoder
        elif 'abmil' in model_name:
            enc = ABMILSlideEncoder
        else:
            raise ValueError(f"Model type {model_name} not supported")
        
        return enc(model_name = model_name, pretrained = pretrained, freeze = freeze, **kwargs)
    
# Map from slide encoder to required patch encoder
# Used in Processor.py to load the correct patch encoder for a given slide encoder
slide_to_patch_encoder_name = {
    'threads': 'conch_v15',
    'titan': 'conch_v15',
    'tcga': 'conch_v15',
    'prism': 'virchow',
    'chief': 'ctranspath',
    'gigapath': 'gigapath'
}

####################################################################################################

class BaseSlideEncoder(torch.nn.Module):
    
    def __init__(self, freeze=True, **build_kwargs):
        '''
        Parent class for all pretrained slide encoders.
        '''
        super().__init__()
        self.enc_name = None
        self.model, self.precision, self.embedding_dim = self._build(**build_kwargs)

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
        '''
        Initialization method, must be defined in child class.
        '''
        pass


####################################################################################################


class ABMILSlideEncoder(BaseSlideEncoder):
    
    def _build(self, input_feature_dim, n_heads, head_dim, dropout, gated, pretrained=False, **kwargs):
        from trident.slide_encoder_models.model_zoo.reusable_blocks.ABMIL import ABMIL
        import torch.nn as nn
        
        assert pretrained is False, "ABMILSlideEncoder has no corresponding pretrained models. Please load with pretrained=False."
                                
        pre_attention_layers = nn.Sequential(
            nn.Linear(input_feature_dim, input_feature_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        image_pooler = ABMIL(
            n_heads=n_heads,
            feature_dim=input_feature_dim,
            head_dim=head_dim,
            dropout=dropout,
            n_branches=1,
            gated=gated
        )
        
        post_attention_layers = nn.Sequential(
            nn.Linear(input_feature_dim, input_feature_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        model = nn.ModuleDict({
            'pre_attention_layers': pre_attention_layers,
            'image_pooler': image_pooler,
            'post_attention_layers': post_attention_layers
        })
        
        precision = torch.float32
        embedding_dim = input_feature_dim
        return model, precision, embedding_dim

    def forward(self, batch, device='cuda'):
        image_features = self.model['pre_attention_layers'](batch['features'].to(device))
        image_features, attn = self.model['image_pooler'](image_features) # Features shape: (b n_branches f), where n_branches = 1. Branching is not used in this implementation.
        image_features = rearrange(image_features, 'b 1 f -> b f')
        image_features = self.model['post_attention_layers'](image_features)# Attention scores shape: (b r h n), where h is number of attention heads 
        return image_features


class PRISMSlideEncoder(BaseSlideEncoder):

    def _build(self, pretrained=True, **kwargs):
        
        self.enc_name = 'prism'

        if sys.version_info < (3, 10):
            raise RuntimeError("PRISM requires Python 3.10 or above. Please update your Python interpreter.")

        try:
            import environs  # weird dependencies required by PRISM
            import sacremoses
            from transformers import AutoModel, AutoConfig
        except:
            traceback.print_exc()
            raise Exception(
                "Please run `pip install environs==11.0.0 transformers==4.42.4 sacremoses==0.1.1` "
                "and ensure Python version is 3.10 or above."
            )

        if pretrained:
            model = AutoModel.from_pretrained('paige-ai/Prism', trust_remote_code=True)
        else:
            model = AutoModel.from_config(AutoConfig.from_pretrained('paige-ai/Prism'))
        model.text_decoder = None
        precision = torch.float16
        embedding_dim = 1280
        return model, precision, embedding_dim
    
    def forward(self, batch, device='cuda'):
        # input should be of shape (batch_size, tile_seq_len, tile_embed_dim)
        x = batch['features'].to(device)
        z = self.model.slide_representations(x)
        z = z['image_embedding'] 
        return z
    

class CHIEFSlideEncoder(BaseSlideEncoder):

    def _build(self, pretrained=True, **kwargs):
        
        self.enc_name = 'chief'
        weights_path = get_weights_path('slide', self.enc_name)

        try:
            import sys; sys.path.append(weights_path)
            from models.CHIEF import CHIEF
        except:
            traceback.print_exc()
            raise Exception(f"Problem importing CHIEF repo from {weights_path}. Please clone CHIEF from https://github.com/hms-dbmi/CHIEF/ and set the path in load_ckpts.json. Also, please check that CHIEF dependencies are installed.")

        current_wd = os.getcwd() # Get current working directory
        os.chdir(weights_path)  # Set working directory to CHIEF repo
        assert os.path.exists(os.path.join(weights_path, './model_weight', 'Text_emdding.pth')), f"Please copy Text_emdding.pth to {os.path.join(weights_path, './model_weight')}. Checkpoint can be downloaded from https://drive.google.com/drive/folders/1uRv9A1HuTW5m_pJoyMzdN31bE1i-tDaV"
        assert os.path.exists(os.path.join(weights_path, './model_weight', 'CHIEF_pretraining.pth')), f"Please copy CHIEF_pretraining.pth to {os.path.join(weights_path, './model_weight')}. Checkpoint can be downloaded from https://drive.google.com/drive/folders/1uRv9A1HuTW5m_pJoyMzdN31bE1i-tDaV"
        model = CHIEF(size_arg="small", dropout=True, n_classes=2)

        # Load pretrained weights
        if pretrained:
            td = torch.load(os.path.join('model_weight', 'CHIEF_pretraining.pth'), map_location='cpu', weights_only=True)
            model.load_state_dict(td, strict=True)
            
        # Return to original working directory
        os.chdir(current_wd)
        
        precision = torch.float32
        embedding_dim = 768
        return model, precision, embedding_dim
    
    def forward(self, batch, device='cuda'):
        x = batch['features'].squeeze(0).to(device)
        z = self.model(x, torch.tensor([0]))
        z = z['WSI_feature']  # Shape (1,768)
        return z
    

class GigaPathSlideEncoder(BaseSlideEncoder):

    def _build(self, pretrained=True, **kwargs):

        self.enc_name = 'gigapath'

        try:
            from gigapath.slide_encoder import create_model
        except:
            traceback.print_exc()
            raise Exception("Please install fairscale and gigapath using `pip install fairscale git+https://github.com/prov-gigapath/prov-gigapath.git`.")
        
        # Make sure flash_attn is correct version
        try:
            import flash_attn; assert flash_attn.__version__ == '2.5.8'
        except:
            traceback.print_exc()
            raise Exception("Please install flash_attn version 2.5.8 using `pip install flash_attn==2.5.8`.")
        
        if pretrained:
            model = create_model("hf_hub:prov-gigapath/prov-gigapath", "gigapath_slide_enc12l768d", 1536, global_pool=True)
        else:
            model = create_model("", "gigapath_slide_enc12l768d", 1536, global_pool=True)
        
        
        precision = torch.float16
        embedding_dim = 768
        return model, precision, embedding_dim

    def forward(self, batch, device='cuda'):
        self.model.tile_size = batch['attributes']['patch_size_level0']
        z = self.model(batch['features'].to(device), batch['coords'].to(device), all_layer_embed=True)[11]
        return z


class ThreadsSlideEncoder(BaseSlideEncoder):

    def _build(self, pretrained=True, **kwargs):

        self.enc_name = 'threads'

        try:
            from threadsmodel.inference import create_model, create_model_from_pretrained
        except:
            traceback.print_exc()
            raise Exception("Coming Soon! Thanks for your patience.")

        return None, None, None

    def forward(self, batch, device='cuda', return_raw_attention=False):
        pass


class TitanSlideEncoder(BaseSlideEncoder):
    
    def _build(self, pretrained=True, **kwargs):
        self.enc_name = 'titan'
        assert pretrained, "TitanSlideEncoder has no non-pretrained models. Please load with pretrained=True."
        from transformers import AutoModel 
        model = AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True)
        precision = torch.float16
        embedding_dim = 768
        return model, precision, embedding_dim

    def forward(self, batch, device='cuda'):
        z = self.model.encode_slide_from_patch_features(batch['features'].to(device), batch['coords'].to(device), batch['attributes']['patch_size_level0'])        
        return z


class MeanSlideEncoder(BaseSlideEncoder):

    def _build(self, model_name = 'mean-default', **kwargs):
        self.enc_name = model_name
        
        if model_name == 'mean-conch_v1':
            embedding_dim = 768
        elif model_name == 'mean-conch_v15':
            embedding_dim = 768
        elif model_name == 'mean-uni_v1':
            embedding_dim = 1024
        elif model_name == 'mean-uni_v2':
            embedding_dim = 1536
        elif model_name == 'mean-ctranspath':
            embedding_dim = 768
        elif model_name == 'mean-phikon':
            embedding_dim = 768
        elif model_name == 'mean-resnet50':
            embedding_dim = 1024
        elif model_name == 'mean-gigapath':
            embedding_dim = 1536
        elif model_name == 'mean-virchow':
            embedding_dim = 2560
        elif model_name == 'mean-virchow2':
            embedding_dim = 2560
        elif model_name == 'mean-hoptimus0':
            embedding_dim = 1536
        elif model_name == 'mean-phikon_v2':
            embedding_dim = 1024
        elif model_name == 'mean-musk':
            embedding_dim = 1024
        else:
            print(f"\033[93mWARNING: Could not automatically infer embedding_dim for mean encoder {self.enc_name}. Setting to None.\033[0m")
            embedding_dim = None
            
        return None, None, embedding_dim

    def forward(self, batch, device='cuda'):
        z = batch['features'].to(device).mean(dim=1) # Just mean pooling
        return z
