import numpy as np
import matplotlib.pyplot as plt
from FootSteps import FootSteps

class ZmpClass(object):
    def __init__(self, footsteps: FootSteps):
        self.footsteps = footsteps     
        self._log_zmp_trajectory()
        
        
    def __call__(self, t):
        foot = self.footsteps.get_phase_type(t)
        left_pos = self.footsteps.get_left_position(t)
        right_pos = self.footsteps.get_right_position(t)
        
        if foot == 'left':
            result = np.array(right_pos)
        elif foot == 'right':
            result = np.array(left_pos)
        else: # double support
            result = np.array([(left_pos[0] + right_pos[0]) / 2, 
                              (left_pos[1] + right_pos[1]) / 2])
        return result
