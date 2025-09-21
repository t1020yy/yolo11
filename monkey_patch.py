from torch import nn
from ultralytics.nn.modules import Proto

_original_proto_init = Proto.__init__

def apply_proto_upsample_patch(downsampling=4):
    """
    replace Proto-Upsample-Layer in Ultralytics YOLO inst seg model.

    Args:
        downsampling (int): Factor for downsampling (1-> maskshape==inputshape, 4 -> maskshape==inputshape/4 )
    """
    org_downsampling = 4
    new_downsampling = downsampling
    org_scale_factor = 2
    new_scale_factor = int((org_downsampling / new_downsampling) * org_scale_factor)
    def custom_proto_init(self, c1, c2, k=1):
        _original_proto_init(self, c1, c2, k)
        self.upsample = nn.ConvTranspose2d(
            c2, c2,
            kernel_size=new_scale_factor,
            stride=new_scale_factor,
            padding=0,
            bias=True
        )
    Proto.__init__ = custom_proto_init
    print(f"[Patch active] Proto.upsample → ConvTranspose2d(kernel={new_scale_factor}, stride={new_scale_factor})")

def undo_proto_patch():
    Proto.__init__ = _original_proto_init
    print("[Patch deactivated] Proto.__init__ reset to original")