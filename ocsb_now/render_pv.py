"""用 pyvista 离屏渲染整房间 mesh 的多视角图。"""
import os

import pyvista as pv

pv.OFF_SCREEN = True
D = 'E:/files/matser_stage/experiment/3D_OCSB/_archive/a8_viz/ply2'
MESH = os.path.join(D, 'room_scene0606_00_mesh.ply')
MESH_C = os.path.join(D, 'room_scene0606_00_mesh_counter.ply')

views = {
    'iso': (1.0, -1.0, 0.75),
    'front': (0.0, -1.3, 0.3),
    'top': (0.02, -0.02, 1.4),
    'back': (-1.0, 1.0, 0.75),
}


def shot(path, eye_dir, out):
    m = pv.read(path)
    print(out, 'points', m.n_points, 'cells', m.n_cells)
    p = pv.Plotter(off_screen=True, window_size=(1800, 1350))
    p.add_mesh(m, rgb=True, smooth_shading=True, show_edges=False)
    p.set_background('white')
    c = m.center
    r = max(m.bounds[1] - m.bounds[0], m.bounds[3] - m.bounds[2],
            m.bounds[5] - m.bounds[4])
    e = [c[i] + r * eye_dir[i] for i in range(3)]
    p.camera_position = [e, c, (0, 0, 1)]
    p.camera.zoom(1.15)
    p.screenshot(out)
    p.close()


for name, d in views.items():
    shot(MESH, d, os.path.join(D, 'room_scene0606_00_%s.png' % name))
shot(MESH_C, (1.0, -1.0, 0.75),
     os.path.join(D, 'room_scene0606_00_counter_iso.png'))
print('done')
