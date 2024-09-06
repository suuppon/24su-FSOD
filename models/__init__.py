from .dynamic_mdetr_resnet import DynamicMDETR as DynamicMDETR_ResNet
from .dynamic_mdetr_clip import DynamicMDETR as DynamicMDETR_CLIP

from .dynamic_fs_mdetr_resnet import DynamicFSMDETR as DynamicFSMDETR_FS_ResNet
# from .dynamic_fs_mdetr_clip import DynamicFSMDETR as DynamicFSMDETR_FS_CLIP

def build_model(args):
    assert args.model_type in ['ResNet', 'CLIP']
    if args.model_type == 'ResNet':
        return DynamicMDETR_ResNet(args)
    elif args.model_type == 'CLIP':
        return DynamicMDETR_CLIP(args)

def build_my_model(args):
    assert args.model_type in ['ResNet', 'CLIP']
    if args.model_type == 'ResNet':
        return DynamicFSMDETR_FS_ResNet(args)
    elif args.model_type == 'CLIP':
        raise NotImplementedError("CLIP model is not supported for the current implementation.")