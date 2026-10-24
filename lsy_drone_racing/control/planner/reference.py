from __future__ import annotations

import numpy as np
import numpy.linalg as la

from scipy.interpolate import BPoly, CubicSpline


class Reference:
    
    DEFAULT_KNOTS = np.array([
        [-1.0, 0.75, 0.4],
        [0.3, 0.35, 0.7],
        [1.3, -0.15, 0.9],
        [0.85, 0.85, 1.2],
        [-0.5, -0.05, 0.7],
        [-1.2, -0.2, 0.8],
        [-1.2, -0.2, 1.2],
        [-0.0, -0.7, 1.2],
        [1.2, -0.15, 1.2],
        [1.05, 0.75, 1.2],
        [0.25, 1.25, 1.2],
    ])
    
    def __init__(
        self,
        sparse_knots,
        tangents=None,
        accs=None,
        ds=0.01
    ):
        
        self.ds = ds
        self._version = 0 
        
        self._update(sparse_knots, tangents, accs)
    
    @classmethod
    def default_reference(cls, obs, **kwargs) -> Reference:
        knots = np.vstack([obs["pos"], cls.DEFAULT_KNOTS])
        return cls(knots, **kwargs)
         
    def _update(self, knots, tangents=None, accs=None):
        
        self._sparse_knots = knots
        if tangents is None and accs is None:
            u = np.append(0, np.cumsum(la.norm(np.diff(self._sparse_knots, axis=0), axis=1)))
            spline = CubicSpline(u, self._sparse_knots, axis=0)

            self._sparse_tangents = spline(u, 1)
            self._sparse_accs = spline(u, 2)
        else:
            self._sparse_tangents = tangents
            self._sparse_accs = accs
            
        self._sparse_curve, u_sparse = self.build_sparse_curve(self._sparse_knots, self._sparse_tangents, self._sparse_accs)
        self.length = u_sparse[-1]
        
        self._dense_knots, u_dense = self.sample_densely()
                
        # reparameterize by progress
        s = np.append(0, np.cumsum(la.norm(np.diff(self._dense_knots, axis=0), axis=1)))
        self.path = CubicSpline(s, self._dense_knots, axis=0)
        
        self.sparse_s = np.interp(u_sparse, u_dense, s)
        self.length = s[-1]
        self._version = self._version + 1
    
    @staticmethod
    def build_sparse_curve(knots, tangents, accs):
        u = np.append(0, np.cumsum(la.norm(np.diff(knots, axis=0), axis=1)))
        if np.any(np.diff(u) <= 0):
            raise ValueError("Consecutive knots must not coincide.")

        curve = BPoly.from_derivatives(u, np.stack([knots, tangents, accs], axis=1)) # (K, 3 derivs, 3 dims)
        return curve, u
    
    def sample_densely(self):
        u =np.linspace(0, self.length, int(self.length / self.ds) + 1)
        return self._sparse_curve(u), u
    
    def update(self, new_knots, new_s, new_tangents=None, new_accs=None, min_spacing=0.3):
        """
        Take old tangents and accs by default
        """
        
        if new_tangents is None and new_accs is None:
            new_tangents = self.path(new_s, 1)
            new_accs = self.path(new_s, 2)
        
        order = np.argsort(new_s)
        new_s = new_s[order]
        new = [new_knots[order], new_tangents[order], new_accs[order]]
        
        idx = np.searchsorted(self.sparse_s, new_s)
        
        left = np.clip(idx - 1, 0, len(self.sparse_s) - 1)
        right = np.clip(idx, 0, len(self.sparse_s) - 1)
        nearest = np.where(np.abs(self.sparse_s[left] - new_s) < np.abs(self.sparse_s[right] - new_s), left, right)
        replace = (np.abs(self.sparse_s[nearest] - new_s) < min_spacing) & (nearest != 0)

        updated = []
        for old, values in zip([self._sparse_knots, self._sparse_tangents, self._sparse_accs], new):
            old = old.copy()
            old[nearest[replace]] = values[replace]
            updated.append(np.insert(old, idx[~replace], values[~replace], axis=0))

        self._update(*tuple(updated))