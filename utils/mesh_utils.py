"""
!!! This code file is not organized, there may be relatively chaotic writing and inconsistent comment formats. !!!
"""

import os
import numpy as np
import trimesh
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R


def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)for g in scene_or_mesh.geometry.values())
            )
    else:
        assert isinstance(scene_or_mesh, trimesh.Trimesh)
        mesh = scene_or_mesh
    return mesh


def extract_colors_from_urdf(urdf_path):
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    global_materials = {}

    for material in root.findall("material"):
        name = material.attrib["name"]
        color_elem = material.find("color")
        if color_elem is not None and "rgba" in color_elem.attrib:
            rgba = [float(c) for c in color_elem.attrib["rgba"].split()]
            global_materials[name] = rgba

    link_colors = {}

    for link in root.iter("link"):
        link_name = link.attrib["name"]
        visual = link.find("./visual")
        if visual is not None:
            material = visual.find("./material")
            if material is not None:
                color = material.find("color")
                if color is not None and "rgba" in color.attrib:
                    rgba = [float(c) for c in color.attrib["rgba"].split()]
                    link_colors[link_name] = rgba
                elif "name" in material.attrib:
                    material_name = material.attrib["name"]
                    if material_name in global_materials:
                        link_colors[link_name] = global_materials[material_name]

    return link_colors


def parse_origin(element):
    """Parse the origin element for translation and rotation."""
    origin = element.find("origin")
    xyz = np.zeros(3)
    rotation = np.eye(3)
    if origin is not None:
        xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
        rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ")
        rotation = R.from_euler("xyz", rpy).as_matrix()
    return xyz, rotation


def apply_transform(mesh, translation, rotation):
    """Apply translation and rotation to a mesh."""
    # mesh.apply_translation(-mesh.centroid)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    mesh.apply_transform(transform)
    return mesh


def create_primitive_mesh(geometry, translation, rotation):
    """Create a trimesh object from primitive geometry definitions with transformations."""
    if geometry.tag.endswith("box"):
        size = np.fromstring(geometry.attrib["size"], sep=" ")
        mesh = trimesh.creation.box(extents=size)
    elif geometry.tag.endswith("sphere"):
        radius = float(geometry.attrib["radius"])
        mesh = trimesh.creation.icosphere(radius=radius)
    elif geometry.tag.endswith("cylinder"):
        radius = float(geometry.attrib["radius"])
        length = float(geometry.attrib["length"])
        mesh = trimesh.creation.cylinder(radius=radius, height=length)
    else:
        raise ValueError(f"Unsupported geometry type: {geometry.tag}")
    return apply_transform(mesh, translation, rotation)


def load_link_geometries(robot_name, urdf_path, link_names, collision=False):
    """Load geometries (trimesh objects) for specified links from a URDF file, considering origins."""
    urdf_dir = os.path.dirname(urdf_path)
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    link_geometries = {}
    link_colors_from_urdf = extract_colors_from_urdf(urdf_path)

    for link in root.findall("link"):
        link_name = link.attrib["name"]
        link_color = link_colors_from_urdf.get(link_name, None)
        if link_name in link_names:
            geom_index = "collision" if collision else "visual"
            link_mesh = []
            for visual in link.findall(".//" + geom_index):
                geometry = visual.find("geometry")
                xyz, rotation = parse_origin(visual)
                try:
                    if geometry[0].tag.endswith("mesh"):
                        mesh_filename = geometry[0].attrib["filename"]
                        full_mesh_path = os.path.join(urdf_dir, mesh_filename)
                        mesh = as_mesh(trimesh.load(full_mesh_path))
                        scale = np.fromstring(geometry[0].attrib.get("scale", "1 1 1"), sep=" ")
                        mesh.apply_scale(scale)
                        mesh = apply_transform(mesh, xyz, rotation)
                        link_mesh.append(mesh)
                    else:  # Handle primitive shapes
                        mesh = create_primitive_mesh(geometry[0], xyz, rotation)
                        scale = np.fromstring(geometry[0].attrib.get("scale", "1 1 1"), sep=" ")
                        mesh.apply_scale(scale)
                        link_mesh.append(mesh)
                except Exception as e:
                    print(f"Failed to load geometry for {link_name}: {e}")
            if len(link_mesh) == 0:
                continue
            elif len(link_mesh) > 1:
                link_trimesh = as_mesh(trimesh.Scene(link_mesh))
            elif len(link_mesh) == 1:
                link_trimesh = link_mesh[0]

            if link_color is not None:
                link_trimesh.visual.face_colors = np.array(link_color)
            link_geometries[link_name] = link_trimesh

    return link_geometries

def _rotation_from_a_to_b(a, b):
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b); c = float(np.dot(a, b)); s = np.linalg.norm(v)
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        axis = np.array([1.,0.,0.]) if abs(a[0]) < 0.9 else np.array([0.,1.,0.])
        v = np.cross(a, axis); v = v / (np.linalg.norm(v) + 1e-12)
        K = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
        return np.eye(3) + 2*K@K  # 180°
    K = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    return np.eye(3) + K + K@K * ((1 - c) / (s**2 + 1e-12))

def add_arrow_mesh(server, name, start, direction, length=0.12, radius=0.003, color=(255,0,0), opacity=1.0):
    start = np.asarray(start, dtype=float)
    d = np.asarray(direction, dtype=float)
    d = d / (np.linalg.norm(d) + 1e-12)

    shaft_len = 0.8 * length
    head_len  = 0.2 * length

    shaft = trimesh.creation.cylinder(radius=radius, height=shaft_len, sections=24)
    head  = trimesh.creation.cone(radius=2.2*radius, height=head_len, sections=24)

    R = _rotation_from_a_to_b(np.array([0,0,1.0]), d)
    T_shaft = np.eye(4); T_shaft[:3,:3] = R
    T_shaft[:3, 3] = start + d * (shaft_len * 0.5)

    T_head = np.eye(4); T_head[:3,:3] = R
    T_head[:3, 3] = start + d * shaft_len

    shaft.apply_transform(T_shaft)
    head.apply_transform(T_head)
    arrow = trimesh.util.concatenate([shaft, head])

    server.scene.add_mesh_simple(
        name,
        arrow.vertices, arrow.faces,
        color=color,
        opacity=opacity
    )
