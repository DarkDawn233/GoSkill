# Src: https://github.com/lucidrains/vector-quantize-pytorch
from .vector_quantize_pytorch import VectorQuantize
from .residual_vq import ResidualVQ, GroupedResidualVQ
from .random_projection_quantizer import RandomProjectionQuantizer
from .finite_scalar_quantization import FSQ
from .lookup_free_quantization import LFQ
from .residual_lfq import ResidualLFQ, GroupedResidualLFQ
from .residual_fsq import ResidualFSQ, GroupedResidualFSQ
from .latent_quantization import LatentQuantize
