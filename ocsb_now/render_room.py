"""渲染整房间 mesh 的四个视角，并导出 counter 高亮版 mesh。

counter 标签从服务器 scenes/data 的 npy 按最近邻映射到 mesh 顶点。
"""
import os

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

MESH = ('E:/files/matser_stage/experiment/3D_OCSB/_archive/a8_viz/ply2/'
        'room_scene0606_00_mesh.ply')
LAB = 'E:/files/matser_stage/experiment/3D_OCSB/ocsb_now/_lab_cache.npz'
OUTDIR = 'E:/files/matser_stage/experiment/3D_OCSB/_archive/a8_viz/ply2'
C = 13

mesh = o3d.io.read_triangle_mesh(MESH)
mesh.compute_vertex_normals()
print('mesh', len(mesh.vertices), len(mesh.triangles))

# counter 高亮版
c = np.load(LAB)
tree = cKDTree(c['xyz'])
_, idx = tree.query(np.asarray(mesh.vertices), k=1)
lab = c['lab'][idx]
col = np.asarray(mesh.vertex_colors).copy()
col[lab == C] = [1.0, 0.0, 1.0]
mesh_hi = o3d.geometry.TriangleMesh(mesh)
mesh_hi.vertex_colors = o3d.utility.Vector3dVector(col)
o3d.io.write_triangle_mesh(
    os.path.join(OUTDIR, 'room_scene0606_00_mesh_counter.ply'),
    mesh_hi, write_ascii=False)
print('counter verts', int((lab == C).sum()))

# 渲染
W, H = 1600, 1200
r = o3d.visualization.rendering.OffscreenRenderer(W, H)
r.scene.set_background([1, 1, 1, 1])
mat = o3d.visualization.rendering.MaterialRecord()
mat.shader = 'defaultLit'
r.scene.add_geometry('mesh', mesh, mat)
bounds = mesh.get_axis_aligned_bounding_box()
center = bounds.get_center()
extent = float(np.linalg.norm(bounds.get_extent()))
views = {
    'iso': center + extent * np.array([0.9, -0.9, 0.7]),
    'front': center + extent * np.array([0.0, -1.2, 0.35]),
    'top': center + extent * np.array([0.0, -0.05, 1.3]),
    'back': center + extent * np.array([-0.9, 0.9, 0.7]),
}
for name, eye in views.items():
    r.setup_camera(60, center, eye, [0, 0, 1])
    img = r.render_to_image()
    p = os.path.join(OUTDIR, 'room_scene0606_00_%s.png' % name)
    o3d.io.write_image(p, img)
    print('->', p)
