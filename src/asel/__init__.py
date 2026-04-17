"""
Affine Steerable Equivariant Layer for Canonicalization of Neural Networks.

See submodules
- `affine`,
- `roto_scale`, and
- `scale`
for the affine equivariant network implementation for the respective group.
"""

from . import EquivarLayer_affine as affine
from . import EquivarLayer_RS as roto_scale
from . import EquivarLayer_scale as scale
