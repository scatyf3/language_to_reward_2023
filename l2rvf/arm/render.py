"""Offscreen rendering helpers (frames, mp4/gif, contact sheets)."""

from __future__ import annotations

import mujoco
import numpy as np


class FrameRecorder:
  def __init__(self, env, every=2, width=480, height=360, azimuth=215,
               elevation=-28, distance=1.35, lookat=(0.45, 0.1, 0.1)):
    self.env = env
    self.every = every
    self.renderer = mujoco.Renderer(env.model, height=height, width=width)
    self.cam = mujoco.MjvCamera()
    self.cam.azimuth, self.cam.elevation = azimuth, elevation
    self.cam.distance, self.cam.lookat[:] = distance, lookat
    self.frames = []
    self._n = 0

  def __call__(self, env):
    if self._n % self.every == 0:
      self.renderer.update_scene(env.data, camera=self.cam)
      self.frames.append(self.renderer.render().copy())
    self._n += 1

  def contact_sheet(self, n=6):
    idx = np.linspace(0, len(self.frames) - 1, n).astype(int)
    return np.concatenate([self.frames[i] for i in idx], axis=1)

  def save_video(self, path, fps=12):
    import imageio.v3 as iio
    iio.imwrite(path, np.stack(self.frames), fps=fps)
