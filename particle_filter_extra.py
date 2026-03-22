"""
EXTRA CREDIT: KLD Sampling

Adaptive particle count using the KLD-sampling algorithm (Fox, 2003).
Instead of resampling a fixed N particles, the resampling loop continues
until the number of particles satisfies:

n ≥ (k-1) / (2ε) x (1 - 2/(9(k-1)) + √(2/(9(k-1))) x z_{1-δ})³

where
    k   = number of non-empty histogram bins in the current sample set
    ε   = 0.05  (allowed KL divergence error)
    δ   = 0.01  → z_{0.99} = 2.326

Bin resolution (position only for simplicity):
    BIN_SIZE_X = 50 px  (1400 / 50 = 28 bins along x)
    BIN_SIZE_Y = 50 px  (800 / 50 = 16 bins along y)
    BIN_ORIENT = 8 discrete orientation sectors

The particle count per step is logged in self.particle_count_log for analysis.

"""

import random
import numpy as np
import bisect
import copy
from utils import add_noise as utils_add_noise
from particle_filter import ParticleFilter, normalize_weights, Particle, WeightedDistribution

# ── KLD-sampling hyper-parameters ────────────────────────────────────────────
KLD_EPSILON        = 0.05    # allowed KL error
KLD_DELTA          = 0.01    # failure probability
KLD_Z              = 2.326   # z_{1 - delta} = z_{0.99}
KLD_MIN_PARTICLES  = 50
KLD_MAX_PARTICLES  = 2000

# ── Histogram bin resolution ─────────────────────────────────────────────────
BIN_SIZE_X  = 50   # pixels per cell (x-axis)
BIN_SIZE_Y  = 50   # pixels per cell (y-axis)
BIN_ORIENT  = 8    # number of orientation sectors



def kld_required_n(k):
    """
    Return the minimum number of particles required given k non-empty bins.

    Formula (Fox 2003):
        n = (k-1) / (2ε) x (1 - 2/(9(k-1)) + √(2/(9(k-1))) x z)³

    Returns at least KLD_MIN_PARTICLES and at most KLD_MAX_PARTICLES.
    """
    if k <= 1:
        return KLD_MIN_PARTICLES
    inner = (1.0
             - 2.0 / (9.0 * (k - 1))
             + np.sqrt(2.0 / (9.0 * (k - 1))) * KLD_Z)
    n = ((k - 1) / (2.0 * KLD_EPSILON)) * (inner ** 3)
    return int(np.clip(np.ceil(n), KLD_MIN_PARTICLES, KLD_MAX_PARTICLES))

def particle_bin(particle):
    """
    Map a particle's (x, y, θ) pose into a discrete 3-D histogram bin.

    Returns a tuple (bx, by, bo) identifying the bin.
    """
    bx = int(particle.pos[0] / BIN_SIZE_X)
    by = int(particle.pos[1] / BIN_SIZE_Y)
    # Map orientation angle [-π, π] to one of BIN_ORIENT sectors
    angle = np.arctan2(particle.orient[1], particle.orient[0])
    bo = int((angle + np.pi) / (2.0 * np.pi) * BIN_ORIENT) % BIN_ORIENT
    return (bx, by, bo)



class ParticleFilterExtra(ParticleFilter):

    def __init__(self, num_particles, minx, maxx, miny, maxy, noise_type="gaussian"):
        super().__init__(num_particles, minx, maxx, miny, maxy, noise_type)
        self.particle_count_log = []


    def filtering(self, sensor, max_sensor_range, sensor_std,
                  evidence, delta_angle, speed):
        """
        One step of particle filtering with KLD-adaptive resampling.

        Steps:
          1. Transition sample  (same as baseline)
          2. Compute & normalise weights  (same as baseline)
          3. KLD-adaptive resampling  (replaces fixed-size resample)
        """
        new_particles = []

        # Step 1: transition
        for particle in self.particles:
            new_particles.append(
                self.transition_sample(particle, delta_angle, speed))

        # Step 2: weights
        for particle in new_particles:
            particle.weight = self.compute_prenorm_weight(
                particle, sensor, max_sensor_range, sensor_std, evidence)
        normalize_weights(new_particles)

        # Step 3: KLD-adaptive resample
        new_particles = self.kld_resample(new_particles)

        # Track particle count for analysis
        self.num_particles = len(new_particles)
        self.particle_count_log.append(self.num_particles)

        return new_particles

    # ── KLD-adaptive resampling ───────────────────────────────────────────
    def kld_resample(self, particles):
        """
        Draw particles one at a time; after each draw, update the bin histogram
        and recompute the required sample size n via kld_required_n(k).
        Stop when len(new_particles) >= n (and >= KLD_MIN_PARTICLES).

        """
        # BEGIN_YOUR_CODE ######################################################
        raise NotImplementedError



        # END_YOUR_CODE ########################################################
        return new_particles