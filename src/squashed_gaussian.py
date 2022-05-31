import math
import torch.nn.functional as F
from torch import distributions as pd
from torch.distributions.transforms import TanhTransform


class SquashedGaussian(pd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale, validate_args=None):
        base_dist = pd.Normal(loc, scale)
        super().__init__(base_dist, TanhTransform(cache_size=1), validate_args=validate_args)

    @property
    def mean(self):
        mu = self.base_dist.loc
        for transform in self.transforms:
            mu = transform(mu)
        return mu