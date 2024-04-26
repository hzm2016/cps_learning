import numpy as np
import matplotlib.pyplot as plt
from overrides import overrides
from src.ll4ma_movement_primitives.phase_variables import ExponentialPV


class ContactExponentialPV(ExponentialPV):
    """
    Implements contact-aware exponential phase variable.

    The phase value is computed in the same way as ExponentialPV,
    but it incorporates feedback such that the phase variable can
    halt as a function of contact forces. See [1] for technical 
    details.

    [1] Conkey, A., & Hermans, T. (2018). Learning Task Constraints 
        from Demonstration for Hybrid Force/Position Control. 
        arXiv preprint arXiv:1811.03026.
    """

    def __init__(self, gamma=2.0, tau=1.0, alpha_f=0.0):
        super(ContactEPV, self).__init__(gamma, tau)        
        self.alpha_f = alpha_f
        self.reset()

    @overrides
    def get_value(self, dt, wrench_mag=None, in_contact=0):
        value = self.x
        feedback = 0.0 if wrench_mag is None else self.alpha_f * wrench_mag
        self.x += self.x * dt * (-self.gamma / (self.tau * (1.0 + feedback)))
        return value

    @overrides
    def reset(self):
        self.x = 1.0
