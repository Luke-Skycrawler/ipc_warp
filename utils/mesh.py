# import meshio
import warp as wp
import numpy as np
import igl
# from pxr import UsdGeom, Usd
# import pyHouGeoIO

class RawMeshFromFile:
    '''
    load mesh from file. 
    Supported formats: obj, usd, bgeo
    Supported primitives: sphere, cylinder
    return format: (#V x 3), (#F x 3), (#V,) for radius
    
    Attributes:
        V: vertex positions (#V x 3)
        F: face indices (#F x 3)
        E: edges
        R: per-vertex radius (#V,) - thickness/radius for each vertex
    '''
    _thickness = 0.01  # Default thickness
    
    @staticmethod
    def load_usd_mesh(filename, usd_path):
        asset_stage = Usd.Stage.Open(filename)
        mesh_geom = UsdGeom.Mesh(asset_stage.GetPrimAtPath(usd_path))

        points = np.array(mesh_geom.GetPointsAttr().Get())
        indices = np.array(mesh_geom.GetFaceVertexIndicesAttr().Get())

        return points, indices

    @staticmethod
    def load_bgeo_triangle_mesh(filename, path):
        mesh = pyHouGeoIO.HTriangleMesh()
        
        pyHouGeoIO.ImportHouGeo(path + filename, mesh)
        # points, indices = wp.load_bgeo_mesh(filename, path)
        # points = np.zeros((0, 3), dtype = np.float32)
        indices = np.zeros((0, 3), dtype = int)
        
        points = mesh.GetPointAttributeMatXf("P")
        indices = mesh.GetTriangleTopology().T
        return points, indices
    def __init__(self, file = "assets/stanford-bunny.obj", folder = "", usd_path = "/Cube", geometry_params = None):
        self.V = np.zeros((0, 3))
        self.F = np.zeros((0, 3), dtype = int)
        vertices, indices = self.V, self.F
        edges = None
        radius_values = None
        
        if file == "sphere":
            '''predefined sphere'''
            params = geometry_params or {}
            radius = params.get("radius", 1.0)
            vertices = np.zeros((1, 3), dtype = float)
            radius_values = np.full(len(vertices), radius)
        elif file == "cylinder":
            '''predefined cylinder'''
            params = geometry_params or {}
            radius = params.get("radius", 1.0)
            # length = params.get("length", 1.0)
            x1 = params.get("x1", [0.0, 0.0, 0.0])
            x2 = params.get("x2", [0.0, 0.0, 1.0])

            vertices, edges = np.array([x1, x2], dtype = float), np.array([[0, 1]], dtype = int)
            radius_values = np.full(len(vertices), radius)
        elif file.endswith((".usd", ".usda", "usdz")):
            '''load usd'''
            vertices, indices = self.load_usd_mesh(folder + file, usd_path)
        elif file.endswith((".bgeo")):
            vertices, indices = self.load_bgeo_triangle_mesh(file, folder)
        elif file.endswith((".obj")):
            '''load obj'''
            vertices, indices = igl.read_triangle_mesh(folder + file)
            indices = np.array(indices, dtype = int).reshape(-1, 3)
            # vertices, _, _, indices, _, _ = igl.read_obj("my_model.obj")
        elif file.endswith((".ma")): 
            mesh = SlabMesh(file)
            vertices, edges, indices, radius_values = mesh.V, mesh.E, mesh.F, mesh.R
        elif file.endswith("tobj"):
            vertices, tets = import_tobj(folder + file)
        self.V, self.F = vertices, indices
        if edges is None: 
            if self.F.shape[0] > 0:
                self.E = igl.edges(self.F)
            else: 
                self.E = np.zeros((0, 2), dtype = int)
        else: 
            self.E = edges
        
        # Initialize per-vertex radius
        if radius_values is None:
            self.R = np.full(len(vertices), self._thickness)
        else:
            self.R = radius_values
        
        self.file = file

if __name__ == "__main__":
    files = ["box.bgeo", "box.obj", "cube.usda"]
    for file in files:
        raw_mesh = RawMeshFromFile(file, "assets/")
        print(file, raw_mesh.V.shape, raw_mesh.F.shape)