import numpy as np
import polyscope as ps
import warp as wp
from geometry.static_scene import StaticScene
from time_integrator_base import *
from dynamic_contacts import *
from viewer import PSViewer


def drape():
    # The interactive example remains here until its boundary attachments are
    # separated into a dedicated example module.
    rod = RodComplexBC(h, meshes=["assets/bar2.tobj"], transforms=[np.eye(4)])
    viewer = PSViewer(rod)
    ps.set_user_callback(viewer.callback)
    ps.set_ground_plane_mode("none")
    ps.show()


if __name__ == "__main__":
    ps.init()
    wp.init()
    drape()
