
class ContactDMPConfig(DMPConfig):

    def __init__(self, **kwargs):
        # TODO document what these are when stable
        super(ContactDMPConfig, self).__init__(**kwargs)
        self.alpha_c  = kwargs.get("alpha_c", 0.0)
        self.alpha_nc = kwargs.get("alpha_nc", 0.0)
        self.alpha_p  = kwargs.get("alpha_p", 0.0)
        self.alpha_f  = kwargs.get("alpha_f", 0.0)
        
