class ProMPConfig:
    """
    This is a configuration class that simplifies ProMP creation.
    """

    def __init__(self, **kwargs):
        """
        Keyword Args:
            num_bfs             (int)     : Number of basis functions
            regr_alpha          (flt)     : Regularization parameter for linear regression
            mu_w_mle            (ndarray) : Sample mean of weight vector for basis functions
            mu_w_conditioned    (ndarray) : Mean of distribution conditioned on waypoints
            sigma_w_mle         (ndarray) : Sample covariance matrix for basis function weights
            sigma_w_prior       (ndarray) : Prior covariance matrix for basis function weights
            sigma_w_conditioned (ndarray) : Covariance of distribution conditioned on waypoints
            inv_wish_factor     (flt)     : Inverse Wishart prior trade-off factor
            dist_threshold      (flt)     : Similarity distance cutoff
            name                (str)     : Reference name for ProMP system
            state_types         (lst)     : List of ProMP state types (e.g. ['q'] for joint positions)
            dimensions          (lst)     : List of lists of dimensions for corresponding ProMP states 
                                            (e.g. [[1,2], [1]] for corresonding state list ['q', 'x'] 
                                            for dimensions 1 and 2 of joint positions and dimension 1 
                                            of pose)
            w_keys              (lst)     : List of weight keys (e.g. ['q.0', 'q.1', 'x.0'])
            num_demos           (flt)     : Number of demonstrations ProMP was learned from
        """
        self.num_bfs = kwargs.get("num_bfs", 10)
        self.regr_alpha = kwargs.get("regr_alpha", 0.5)
        self.mu_w_mle = kwargs.get("mu_w_mle", None)
        self.sigma_w_mle = kwargs.get("sigma_w_mle", None)
        self.sigma_w_prior = kwargs.get("sigma_w_prior", None)
        self.inv_wish_factor = kwargs.get("inv_wish_factor", 0.0)
        self.dist_threshold = kwargs.get("dist_threshold", 0.0)
        self.name = kwargs.get("name", "DEFAULT")
        self.state_types = kwargs.get("state_types", [])
        self.dimensions = kwargs.get("dimensions", [])  
        self.w_keys = kwargs.get("w_keys", [])  
        self.num_demos = kwargs.get("num_demos", 0.0)  
        self.num_dims = kwargs.get("num_dims", 0)  
        self.init_joint_state = kwargs.get("init_joint_state", [])
        self.fixed_mahalanobis_threshold = kwargs.get(
            "fixed_mahalanobis_threshold", 12.0)  
