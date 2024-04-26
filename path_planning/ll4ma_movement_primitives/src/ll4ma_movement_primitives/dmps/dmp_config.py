
class DMPConfig(object):
    """
    Configuration object for initializing DMPs in a concise and
    controllable manner.
    """
    
    def __init__(self, **kwargs):
        """
        Keyword Args:
            init       (flt)    : DMP Starting state
            goal       (flt)    : DMP Goal state
            num_bfs    (int)    : Number of basis functions
            dt         (flt)    : Timestep
            tau        (flt)    : Temporal scaling factor (~ trajectory duration)
            gamma      (flt)    : Phase variable scaling factor
            alpha      (flt)    : Proportional gain for DMP system
            beta       (flt)    : Derivative gain for DMP system
            regr_alpha (flt)    : Regularization parameter for linear regression
            w          (ndarray): Weight vector for basis functions
            name       (str)    : Reference name for DMP system
            state_type (str)    : Type of DMP state (e.g. 'q' for joint positions)
            dimension  (int)    : Dimension of DMP state (e.g. 0 for first joint position)
        """
        self.init       = kwargs.get("init", None)
        self.goal       = kwargs.get("goal", None)
        self.num_bfs    = kwargs.get("num_bfs", 10)
        self.dt         = kwargs.get("dt", -1)
        self.tau        = kwargs.get("tau", 5.0)
        self.gamma      = kwargs.get("gamma", 2.0)
        self.alpha      = kwargs.get("alpha", 100.0)
        self.beta       = kwargs.get("beta", 40.0)
        self.regr_alpha = kwargs.get("regr_alpha", 0.5)
        self.w          = kwargs.get("w", None)
        self.name       = kwargs.get("name", "default")
        self.state_type = kwargs.get("state_type", "unknown")
        self.dimension  = kwargs.get("dimension", "unknown")
