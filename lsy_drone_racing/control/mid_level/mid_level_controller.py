import numpy as np
import numpy.linalg as la
from lsy_drone_racing.control.planner.reference import Reference


class MidLevelController:
    
    def __init__(self, 
                 planner,
                 v_max, a_long_max, a_lat_max, a_brake_max,
                 progress_window=(0.2, 0.2)):
        
        self._planner = planner
        
        self.v_max = v_max
        self.a_long_max = a_long_max
        self.a_lat_max = a_lat_max
        self.a_brake_max = a_brake_max
        # (behind, ahead) [m] window around the predicted progress in which the path is searched
        self.progress_window = progress_window
        
        self._current_reference_version = 0
        self._current_velocity_profile = None
        
        self._current_progress = 0.0
    
    def get_trajectory(self, obs, horizon, dt):
        
        ref = self._planner.plan(obs)
        path = ref.path
        
        if self._current_reference_version != ref._version:
            self._current_velocity_profile = self._compute_velocity_profile(ref)
            self._current_reference_version = ref._version

        s_k = np.zeros((horizon + 1,))
        v_k = np.zeros((horizon + 1,))

        current_speed = la.norm(obs["vel"])
        # predict progress by integrating the velocity component along the path tangent, then
        # correct it by projecting onto the path in a small window around the prediction
        tangent = path(self._current_progress, 1)
        tangent = tangent / max(la.norm(tangent), 1e-9)
        s_pred = np.clip(self._current_progress + dt * np.dot(obs["vel"], tangent), 0.0, ref.length)
        s0 = ref.get_progress(obs["pos"], prev_progress=s_pred, search_range_s=self.progress_window)

        s_k[0] = s0
        v_k[0] = current_speed

        def velocity_profile(progress):

            i = np.clip(np.searchsorted(ref.path.x, progress), 0, len(ref.path.x) - 1)

            return self._current_velocity_profile[i]
    
        for k in range(horizon):
            v_k[k + 1] = min(velocity_profile(s_k[k]), v_k[k] + self.a_long_max * dt)
            s_k[k + 1] = s_k[k] + v_k[k] * dt
        
        ref_window = np.hstack([path(s_k), path(s_k, 1) * v_k[:, None]])
        
        self._current_progress = s0
        return ref_window # (H + 1, 6)
    
    def _compute_velocity_profile(self, ref: Reference, eps=1e-6):
        # our path is parameterized by arc length
        s = ref.path.x 
        # therefore the second derivative is curvature kappa
        kappa = la.norm(ref.path(s, 2), axis=-1)
     
        # kappa: 1 / R
        # a_lat = v^2 / R
        # => v = sqrt(a_lat / kappa)
        v = np.minimum(self.v_max, np.sqrt(np.maximum(self.a_lat_max / kappa, eps)))
        ds = np.diff(s)
        for i in range(len(v) - 1): # cap at max acceleration
            v[i + 1] = min(v[i + 1], np.sqrt(v[i] ** 2 + 2 * self.a_long_max * ds[i]))
            
        for i in range(len(v) - 2, -1, -1):
            v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * self.a_brake_max * ds[i]))
        
        return v