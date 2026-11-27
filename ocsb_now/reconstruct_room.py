"""用本地 ScanNet200 的 coord/color/normal 做 Poisson 表面重建（给人看的实心房间）。"""
import os
import time

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

D = 'E:/files/matser_stage/experiment/dataset/ScanNet200/val/scene0606_00'
OUT = ('E:/files/matser_stage/experiment/3D_OCSB/_archive/a8_viz/ply2/'
       'room_scene0606_00_mesh.ply')

coord = np.load(os.path.join(D, 'coord.npy')).astype(np.float64)
color = np.load(os.path.join(D, 'color.npy')).astype(np.float64) / 255.0
normal = np.load(os.path.join(D, 'normal.npy')).astype(np.float64)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(coord)
pcd.colors = o3d.utility.Vector3dVector(color)
pcd.normals = o3d.utility.Vector3dVector(normal)
print('points', len(pcd.points), flush=True)

t0 = time.time()
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd, depth=9)
print('poisson %.1fs  verts %d  faces %d'
      % (time.time() - t0, len(mesh.vertices), len(mesh.triangles)), flush=True)

densities = np.asarray(densities)
mesh.remove_vertices_by_mask(densities < np.quantile(densities, 0.03))
mesh.compute_vertex_normals()
print('after trim verts %d  faces %d'
      % (len(mesh.vertices), len(mesh.triangles)), flush=True)

tree = cKDTree(coord)
_, idx = tree.query(np.asarray(mesh.vertices), k=1)
mesh.vertex_colors = o3d.utility.Vector3dVector(color[idx])

o3d.io.write_triangle_mesh(OUT, mesh, write_ascii=False)
print('-> %s  %.1f MB' % (OUT, os.path.getsize(OUT) / 1e6))
