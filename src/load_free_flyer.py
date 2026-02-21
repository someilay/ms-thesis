import mujoco
import xml.etree.ElementTree as ET
from pathlib import Path


def load_urdf(
    model_folder_path: Path, 
    filename: str,
    fixed: bool = False,
    add_floor: bool = True,
    collision_free: bool = True,
    floor_z: float | None = None,
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model_path = model_folder_path / filename
    name = model_path.stem

    # Parse initial urdf file, and save assets
    with open(model_path, "r") as file:
        urdf_string = file.read()
    root = ET.fromstring(urdf_string)
    assets = {}
    for mesh in root.iter("mesh"):
        mesh_path = mesh.get("filename")
        if mesh_path is None:
            continue
        mesh_path = Path(mesh_path)
        with open(model_folder_path / mesh_path, "rb") as f:
            assets[mesh_path.name] = f.read()

    # Load URDF and convert to MuJoCo XML
    model_temp = mujoco.MjModel.from_xml_path(model_path.as_posix(), assets)
    converted_xml_path = model_folder_path / f"{name}.xml"
    mujoco.mj_saveLastXML(converted_xml_path.as_posix(), model_temp)

    # Parse and modify XML to add ground plane
    with open(converted_xml_path, "r") as file:
        xml_string = file.read()
    root = ET.fromstring(xml_string)

    # Find or create worldbody
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("World body not found")

    # Find or create asset
    asset = root.find("asset")
    if asset is None:
        raise ValueError("asset not found")

    # Convert to free-flyer: wrap all robot elements in a body with free joint
    # Collect all existing direct children of worldbody (except what we'll add)
    robot_elements = []
    for child in list(worldbody):
        # Don't include elements we're about to add
        robot_elements.append(child)

    # Clear worldbody
    worldbody.clear()

    # Create a new root body for the robot with free joint
    root_body = ET.Element("body", attrib={"name": "robot_root", "pos": "0 0 0"})
    if not fixed:
        root_body_joint = ET.Element("freejoint", attrib={"name": "root_freejoint"})
        root_body.append(root_body_joint)

    # Add all robot elements to the root body
    for elem in robot_elements:
        root_body.append(elem)

    # Add light at the beginning
    light = ET.Element(
        "light",
        attrib={
            "pos": "0 0 1.5",
            "dir": "0 0 -1",
            "directional": "true",
        },
    )
    worldbody.append(light)

    # Add ground plane
    ground = ET.Element(
        "geom",
        attrib={
            "name": "floor",
            "type": "plane",
            "size": "0 0 0.05",
            "material": "groundplane",
            "pos": f"0 0 {floor_z or '0'}",
        },
    )
    if add_floor:
        worldbody.append(ground)

    # Add the robot with free joint
    worldbody.append(root_body)

    # Add sky box
    skybox_texture = ET.Element(
        "texture",
        attrib={
            "type": "skybox",
            "builtin": "gradient",
            "rgb1": "0.3 0.5 0.7",
            "rgb2": "0 0 0",
            "width": "512",
            "height": "3072",
        },
    )
    asset.insert(1, skybox_texture)

    # Add ground plane texture
    groundplane_texture = ET.Element(
        "texture",
        attrib={
            "type": "2d",
            "name": "groundplane",
            "builtin": "checker",
            "mark": "edge",
            "rgb1": "0.2 0.3 0.4",
            "rgb2": "0.1 0.2 0.3",
            "markrgb": "0.8 0.8 0.8",
            "width": "300",
            "height": "300",
        },
    )
    if add_floor:
        asset.insert(1, groundplane_texture)

    # Add groundplane material
    groundplane_material = ET.Element(
        "material",
        attrib={
            "name": "groundplane",
            "texture": "groundplane",
            "texuniform": "true",
            "texrepeat": "5 5",
            "reflectance": "0.2",
        },
    )
    if add_floor:
        asset.insert(1, groundplane_material)

    # Disable collision detection for all geoms
    if collision_free:
        for geom in root.iter("geom"):
            geom.set("contype", "0")
            geom.set("conaffinity", "0")

    # Convert back to string and reload
    # ET.indent(root, space="\t")
    modified_xml = ET.tostring(root, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(modified_xml, assets)
    data = mujoco.MjData(model)

    return model, data
