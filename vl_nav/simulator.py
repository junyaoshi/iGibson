from gibson2.simulator import Simulator
from gibson2.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
import pybullet as p


class VLNSimulator(Simulator):
    """
    Simulator class is a wrapper of physics simulator (pybullet) and MeshRenderer, it loads objects into
    both pybullet and also MeshRenderer and syncs the pose of objects and robot parts.
    """

    def __init__(self,
                 gravity=9.8,
                 physics_timestep=1 / 120.0,
                 render_timestep=1 / 30.0,
                 mode='gui',
                 image_width=128,
                 image_height=128,
                 vertical_fov=90,
                 device_idx=0,
                 render_to_tensor=False,
                 rendering_settings=MeshRendererSettings()):
        """
        :param gravity: gravity on z direction.
        :param physics_timestep: timestep of physical simulation, p.stepSimulation()
        :param render_timestep: timestep of rendering, and Simulator.step() function
        :param mode: choose mode from gui, headless, iggui (only open iGibson UI), or pbgui(only open pybullet UI)
        :param image_width: width of the camera image
        :param image_height: height of the camera image
        :param vertical_fov: vertical field of view of the camera image in degrees
        :param device_idx: GPU device index to run rendering on
        :param render_to_tensor: Render to GPU tensors
        :param rendering_settings: rendering setting
        """
        # physics simulator
        super(VLNSimulator, self).__init__(
            gravity=gravity,
            physics_timestep=physics_timestep,
            render_timestep=render_timestep,
            mode=mode,
            image_width=image_width,
            image_height=image_height,
            vertical_fov=vertical_fov,
            device_idx=device_idx,
            render_to_tensor=render_to_tensor,
            rendering_settings=rendering_settings
        )

    def step(self, sync=True):
        """
        Step the simulation at self.render_timestep and update positions in renderer
        """
        for _ in range(int(self.render_timestep / self.physics_timestep)):
            p.stepSimulation()
        if sync:
            self.sync()


