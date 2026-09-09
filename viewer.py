import numpy as np
import polyscope as ps
import polyscope.imgui as gui
import igl
from geometry.static_scene import StaticScene

class PSViewer:
    def __init__(self, rod, static_mesh: StaticScene = None):
        self.V = rod.xcs.numpy()
        self.F = rod.F

        self.ps_mesh = ps.register_surface_mesh("rod", self.V, self.F)
        self.frame = 0
        self.rod = rod
        self.ui_pause = True
        self.animate = False
        
        self.end_frame = 5000
        self.capture_interval = 1
        if static_mesh is not None:
            Vs = static_mesh.xcs.numpy()
            Fs = static_mesh.indices.numpy().reshape((-1, 3))
            self.static_mesh = ps.register_surface_mesh("static", Vs, Fs)
            # self.static_mesh.add_vector_quantity("normal", static_mesh.N, defined_on="faces")
            if static_mesh.has_medials:
                self.static_medials = ps.register_curve_network("static medial", static_mesh.V_medial, static_mesh.E_medial)
                self.static_spheres = ps.register_point_cloud("static spheres", static_mesh.V_medial)
                self.static_spheres.add_scalar_quantity("radius", static_mesh.R)
                self.static_spheres.set_point_radius_quantity("radius", autoscale=False)

    def save(self):
        # ps.screenshot(f"output/{self.frame:04d}.jpg")
        self.frame = self.rod.frame
        if self.frame % 4 == 0:
            igl.write_obj(f"output/obj/{self.frame:04d}.obj", self.V, self.F)
        if hasattr(self.rod, "save_states"):
            self.rod.save_states()

    def callback(self):
        changed, self.ui_pause = gui.Checkbox("Pause", self.ui_pause)
        self.animate = gui.Button("Step") or not self.ui_pause
        if gui.Button("Reset"):
            self.rod.reset()
            self.frame = self.rod.frame
            self.ui_pause = True
            self.animate = True

        if gui.Button("Save"):
            np.save(f"output/x_{self.frame}.npy", self.V)
            if hasattr(self.rod, "save_checkpoint"):
                self.rod.save_checkpoint(
                    f"output/checkpoints/frame_{self.rod.frame:04d}.npz"
                )
            print(f"output/x_{self.frame}.npy saved")
        if self.animate: 
            self.rod.step()
            self.V = self.rod.states.x.numpy()
            self.ps_mesh.update_vertex_positions(self.V)
            self.frame = self.rod.frame
            
            print("frame = ", self.frame)

