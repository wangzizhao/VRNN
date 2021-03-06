from distribution.base import distribution


# just used in sampler, so no need to implement log_prob
class dirac_delta(distribution):
    """
    multivariate dirac_delta distribution,
    used for sampling from fhn, lorenz model without noise
    """
    def __init__(self, transformation):
        self.transformation = transformation

    def sample(self, Input):
        return self.transformation.transform(Input)
