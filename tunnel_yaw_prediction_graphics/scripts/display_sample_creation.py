import subt_proc_gen.display_functions as pcdp
from subt_proc_gen.tunnel import TunnelNetwork, TunnelNetworkParams
from subt_proc_gen.mesh_generation import TunnelNetworkMeshGenerator
from subt_proc_gen.tunnel import GrownTunnelGenerationParams
from subt_proc_gen.geometry import Spline3D, Point3D
import numpy as np
import pyvista as pv
from pyvista.plotting.plotting import Plotter as pvPlotter
import logging
from scipy.spatial.transform import Rotation
import os
import trimesh

logging.basicConfig(level=logging.DEBUG)


def gen_hokuyo_rays(width=1024):
    ones = np.ones((1, width))
    theta = np.reshape(np.linspace(np.deg2rad(0), np.deg2rad(0), 1), (1, 1))
    chi = np.reshape(np.linspace(0, 2 * np.pi, width), (1, width))
    thetas = np.reshape(ones * theta, -1)
    chis = np.reshape(ones * chi, -1)
    x = np.cos(thetas) * np.cos(chis)
    y = np.cos(thetas) * np.sin(chis)
    z = np.sin(thetas)
    vectors = np.vstack((x, y, z))
    return vectors


class RayCaster(object):
    def __init__(self, base_rays, mesh: trimesh.Trimesh):
        self.base_rays = base_rays
        self.mesh = mesh
        self.scene = self.mesh.scene()

    def cast_ray(self, camR, cam_t):
        """
        :param cam_rot: (3,) array, camera rotation in Euler angle format.
        :param cam_t: (3,) array, camera translation.
        """
        vectors = (camR @ self.base_rays).T
        origins = np.ones(vectors.shape) * cam_t
        points, index_ray, index_tri = self.mesh.ray.intersects_location(
            origins, vectors, multiple_hits=False
        )
        depth_raw = np.linalg.norm(points - cam_t, 2, 1)
        depth_formatted = np.zeros(len(vectors))
        depth_formatted[index_ray] = depth_raw
        return depth_formatted, points, vectors, origins


def gen_tunnel_network():
    tn = TunnelNetwork(TunnelNetworkParams(max_inclination_rad=np.deg2rad(90)))
    tn.add_random_grown_tunnel(
        params=GrownTunnelGenerationParams(
            distance=100,
            horizontal_tendency_rad=0,
            vertical_tendency_rad=np.deg2rad(0),
            horizontal_noise_rad=np.deg2rad(10),
            vertical_noise_rad=np.deg2rad(10),
            min_segment_length=10,
            max_segment_length=10,
        ),
        n_trials=200000,
    )
    return tn


def gen_mesh(tn: TunnelNetwork):
    tnmg = TunnelNetworkMeshGenerator(tn)
    tnmg.compute_all()
    return tnmg


def get_axis_pose(aps, avs):
    n = np.random.randint(0, len(aps))
    ap = aps[n]
    av = avs[n]
    return ap, av


def v_to_th_ph(av):
    x, y, z = av
    m = np.linalg.norm(av, 2)
    th = np.arcsin(z / m)
    ph = np.arctan2(y, x)
    return th, ph


def ph_th_to_v(ph, th):
    z = np.sin(th)
    x = np.cos(ph)
    y = np.sin(ph)
    return np.array((x, y, z))


def get_perp_vects(v):
    v = np.array(v)
    ath, aph = v_to_th_ph(v)
    nath = ath + np.deg2rad(1)
    naph = aph
    nav = ph_th_to_v(naph, nath)
    v1 = np.cross(nav, v)
    v1 /= np.linalg.norm(v1, 2)
    v2 = np.cross(v, v1)
    return v1, v2


def v_to_R(v):
    v1, v2 = get_perp_vects(v)
    v = v.reshape((3, 1))
    v1 = v1.reshape((3, 1))
    v2 = v2.reshape((3, 1))
    R = Rotation.from_matrix(np.hstack((v, v1, v2))).as_matrix()
    return R


def R_p_to_T(R, p):
    "R must be a 3x3 numpy matrix and p a 3x1 numpy array"
    return np.vstack(
        (
            np.hstack((R, p)),
            np.array(
                (0, 0, 0, 1),
            ),
        ),
    )


def p_v_to_T(ap, av):
    ap = np.array(ap).reshape((3, 1))
    R = v_to_R(av)
    return R_p_to_T(R, ap)


def xyzrpy_to_T(x, y, z, roll, pitch, yaw):
    R = Rotation.from_euler("xyz", (roll, pitch, yaw)).as_matrix()
    pose = np.array((x, y, z))
    pose = pose.reshape((3, 1))
    return R_p_to_T(R, pose)


def plot_T(plotter: pvPlotter, T, color="r", mag=2):
    vx = T[:3, 0]
    vy = T[:3, 1]
    vz = T[:3, 2]
    p = T[:3, 3]
    plotter.add_mesh(pv.PolyData(p), color=color, render_points_as_spheres=True, point_size=3)
    plotter.add_arrows(p, vx, color="r", render_points_as_spheres=True, mag=mag)
    plotter.add_arrows(p, vy, color="g", render_points_as_spheres=True, mag=mag)
    plotter.add_arrows(p, vz, color="b", render_points_as_spheres=True, mag=mag)


def load_or_generate_tn():
    path_to_folder_of_this_file = os.path.dirname(os.path.abspath(__file__))
    path_to_data_folder = os.path.join(path_to_folder_of_this_file, "data")
    os.makedirs(path_to_data_folder, exist_ok=True)
    path_to_mesh = os.path.join(path_to_data_folder, "mesh.obj")
    path_to_aps_avs = os.path.join(path_to_data_folder, "aps_avs.npy")
    path_to_tunnel_nodes = os.path.join(path_to_data_folder, "tnodes.npy")
    if os.path.exists(path_to_mesh) and os.path.exists(path_to_aps_avs):
        with open(path_to_aps_avs, "rb") as f:
            aps_avs = np.load(f)
            aps = aps_avs[:, :3]
            avs = aps_avs[:, 3:6]
        with open(path_to_tunnel_nodes, "rb") as f:
            tnodes = np.load(f)
            tspline = Spline3D([Point3D(node) for node in tnodes])
        mesh = pv.read(path_to_mesh)
        trimesh_mesh = trimesh.load_mesh(path_to_mesh)
    else:
        tn = gen_tunnel_network()
        tnmg = gen_mesh(tn)
        aps = tnmg.aps
        avs = tnmg.avs
        mesh = tnmg.pyvista_mesh
        tunnel = list(tn.tunnels)[0]
        tspline = tunnel.spline
        with open(path_to_aps_avs, "wb+") as f:
            np.save(f, np.hstack((aps, avs)))
        with open(path_to_tunnel_nodes, "wb+") as f:
            np.save(f, np.vstack([n.xyz for n in tunnel.nodes]))
        pv.save_meshio(path_to_mesh, mesh)
        trimesh_mesh = trimesh.load_mesh(path_to_mesh)
    return mesh, tspline, aps, avs, trimesh_mesh


def main():
    pv_mesh, tspline, aps, avs, trimesh_mesh = load_or_generate_tn()
    ap, av = aps[30], avs[30]
    w_to_ap_T = p_v_to_T(ap, av)
    h_trans_T = xyzrpy_to_T(0, 2, 0, 0, 0, 0)
    v_trans_T = xyzrpy_to_T(0, 0, 1, 0, 0, 0)
    yaw_rot_T = xyzrpy_to_T(0, 0, 0, 0, 0, np.deg2rad(-30))
    # rp_rot_T = xyzrpy_to_T(0, 0, 0, np.deg2rad(5), np.deg2rad(-5), 0)
    T1 = w_to_ap_T @ yaw_rot_T
    T2 = T1 @ h_trans_T @ v_trans_T
    # T3 = T2 @ yaw_rot_T
    # T4 = T3 @ rp_rot_T
    caster = RayCaster(gen_hokuyo_rays(), trimesh_mesh)
    depth_formatted, points, vectors, origins = caster.cast_ray(T2[:3, :3], T2[:3, 3].T)
    # Get the axis used to generate the label
    # apls = aps[np.abs(np.linalg.norm(aps[:, :3] - ap, ord=2, axis=1) - 5) < 0.4, :]
    # apls_in_T4 = (np.linalg.inv(T4) @ np.vstack((apls.T, np.ones(len(apls))))).T
    # apls = apls[apls_in_T4[:, 0] > 0]
    # Plot the axis
    # plotter = pvPlotter()
    # mesh_actor = plotter.add_mesh(pv_mesh, style="wireframe")
    # spline_actor = pcdp.plot_spline(plotter, tspline)
    # plot_T(plotter, w_to_ap_T)
    # plotter.show()
    # camera = plotter.camera_position
    # print(camera)
    camera = [
        (0.4592329951803886, -1.8584811447418002, 2.61592577932257),
        (28.517970354543152, 5.131045460058816, -5.590314154980704),
        (0.29713254850024146, -0.07524337853105542, 0.95186694585377),
    ]
    # SCREENSHOTS FROM NOW ON
    # Plot the axis
    plotter = pvPlotter(off_screen=True)
    plotter.set_background("w")
    mesh_actor = plotter.add_mesh(pv_mesh, color="k", style="wireframe")
    spline_actor = pcdp.plot_spline(plotter, tspline, radius=0.05, color="purple")
    plot_T(plotter, w_to_ap_T)
    plotter.camera_position = camera
    plotter.show(screenshot="/home/lorenzo/images/papers/subt_proc_gen/use_case_1.png")
    # Plot the hdisp
    plotter = pvPlotter(off_screen=True)
    plotter.set_background("w")
    plotter.add_actor(mesh_actor)
    plotter.add_actor(spline_actor)
    plot_T(plotter, T1)
    plotter.camera_position = camera
    plotter.show(screenshot="/home/lorenzo/images/papers/subt_proc_gen/use_case_2.png")
    # Plot the vdisp
    plotter = pvPlotter(off_screen=True)
    plotter.set_background("w")
    plotter.add_actor(mesh_actor)
    plotter.add_actor(spline_actor)
    plot_T(plotter, T2)
    # plotter.add_mesh(pv.PolyData(points).glyph(pv.Sphere(radius=0.1)), color="orange")
    plotter.camera_position = camera
    plotter.show(screenshot="/home/lorenzo/images/papers/subt_proc_gen/use_case_3.png")
    # Plot the yaw
    plotter = pvPlotter(off_screen=True)
    plotter.set_background("w")
    plotter.add_actor(mesh_actor)
    plotter.add_actor(spline_actor)
    plot_T(plotter, T2)
    plotter.add_mesh(pv.PolyData(points).glyph(pv.Sphere(radius=0.1)), color="orange")
    plotter.camera_position = camera
    plotter.show(screenshot="/home/lorenzo/images/papers/subt_proc_gen/use_case_4.png")
    exit()
    # Plot the roll and pitch
    plotter = pvPlotter(off_screen=True)
    plotter.set_background("w")
    plotter.add_actor(mesh_actor)
    plotter.add_actor(spline_actor)
    plot_T(plotter, T4)
    plotter.camera_position = camera
    plotter.show(screenshot="/home/lorenzo/images/papers/subt_proc_gen/use_case_5.png")
    # Plot the label axis points
    plotter = pvPlotter(off_screen=True)
    plotter.set_background("w")
    plotter.add_actor(mesh_actor)
    plotter.add_actor(spline_actor)
    plot_T(plotter, T4)
    sphere = pv.Sphere(radius=0.2)
    plotter.add_mesh(
        pv.PolyData(apls).glyph(geom=sphere),
        color="pink",
    )
    plotter.add_mesh(pv.Tube(apls[0, :], T4[:3, 3].T, radius=0.06), color="pink")
    plotter.camera_position = camera
    plotter.show(screenshot="/home/lorenzo/images/papers/subt_proc_gen/use_case_6.png")
    # Plot the points
    plotter = pvPlotter(off_screen=True)
    plotter.set_background("w")
    plotter.add_actor(mesh_actor)
    plotter.add_actor(spline_actor)
    plot_T(plotter, T4)
    plotter.add_mesh(pv.PolyData(points).glyph(pv.Sphere(radius=0.1)), color="orange")
    plotter.camera_position = camera
    plotter.show(screenshot="/home/lorenzo/images/papers/subt_proc_gen/use_case_7.png")


if __name__ == "__main__":
    main()
