# Converts one FBX scene to glTF 2.0 (separate .gltf + .bin, no images) with
# Blender and writes blender-report.json next to it: object/mesh/triangle
# counts (unique and instance-expanded), units and bounds, every material's
# image-texture wiring with absolute file paths, lights, and the scene camera
# path sampled per frame (glTF Y-up space). Normals and MikkTSpace tangents of
# the active UV map are exported (the geometry builder requires TANGENT for
# normal-mapped primitives). Materials are rewritten afterwards
# by scripts/apply-fbx-material-map.mjs; this script decides nothing about them.
#
#   blender -b --factory-startup -Y --python scripts/blender/convert-fbx-scene.py -- \
#     --input scene.fbx --output new-directory [--camera-samples 4096]
#
# The FBX is untrusted data read only by Blender's importer (-Y disables
# auto-run scripts, --factory-startup ignores user add-ons and preferences).
# Output goes to a new directory; an existing directory is refused.
import bpy
import hashlib
import json
import math
import os
import sys
import time


def fail(code):
    print('CONVERT_FBX_ERROR ' + code, file=sys.stderr)
    sys.exit(2)


def arguments():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    values = {}
    i = 0
    while i < len(argv):
        key = argv[i]
        if key not in ('--input', '--output', '--camera-samples') or key in values or i + 1 >= len(argv):
            fail('ARGUMENTS')
        values[key] = argv[i + 1]
        i += 2
    if '--input' not in values or '--output' not in values:
        fail('USAGE: --input scene.fbx --output new-directory [--camera-samples n]')
    samples = int(values.get('--camera-samples', '4096'))
    if samples < 0 or samples > 100000:
        fail('CAMERA_SAMPLES')
    return os.path.realpath(values['--input']), os.path.abspath(values['--output']), samples


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b''):
            digest.update(chunk)
    return digest.hexdigest()


def to_gltf(v):
    # Blender is Z-up right-handed; glTF is Y-up right-handed: (x, y, z) -> (x, z, -y).
    return [v[0], v[2], -v[1]]


def rounded(values, digits=6):
    return [round(float(v), digits) for v in values]


def main():
    source, output, camera_samples = arguments()
    if not source.lower().endswith('.fbx') or not os.path.isfile(source):
        fail('INPUT')
    if os.path.exists(output):
        fail('OUTPUT_EXISTS')
    started = time.time()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    result = bpy.ops.import_scene.fbx(filepath=source, use_custom_normals=True, use_image_search=False, use_anim=True)
    if 'FINISHED' not in result:
        fail('IMPORT_FAILED')
    import_seconds = time.time() - started
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()

    types = {}
    for obj in scene.objects:
        types[obj.type] = types.get(obj.type, 0) + 1
    users = {}
    for obj in scene.objects:
        if obj.type == 'MESH':
            users[obj.data.name] = users.get(obj.data.name, 0) + 1
    meshes = []
    unique_triangles = 0
    expanded_triangles = 0
    largest = []
    for mesh in bpy.data.meshes:
        mesh.calc_loop_triangles()
        triangles = len(mesh.loop_triangles)
        unique_triangles += triangles
        expanded_triangles += triangles * users.get(mesh.name, 0)
        largest.append((triangles, mesh.name))
        meshes.append({'name': mesh.name, 'triangles': triangles, 'vertices': len(mesh.vertices),
                       'users': users.get(mesh.name, 0), 'materials': [m.name if m else None for m in mesh.materials],
                       'uvLayers': [layer.name for layer in mesh.uv_layers], 'activeUv': mesh.uv_layers.active_index,
                       'activeRenderUv': next((i for i, layer in enumerate(mesh.uv_layers) if layer.active_render), None),
                       'colorAttributes': [a.name for a in mesh.color_attributes]})
    largest.sort(reverse=True)

    lo = [math.inf] * 3
    hi = [-math.inf] * 3
    for obj in scene.objects:
        if obj.type != 'MESH':
            continue
        for corner in obj.bound_box:
            world = obj.matrix_world @ __import__('mathutils').Vector(corner)
            for axis in range(3):
                lo[axis] = min(lo[axis], world[axis])
                hi[axis] = max(hi[axis], world[axis])

    materials = {}
    for material in bpy.data.materials:
        entry = {'blendMethod': getattr(material, 'blend_method', None),
                 'surfaceRenderMethod': getattr(material, 'surface_render_method', None),
                 'useBackfaceCulling': bool(material.use_backface_culling), 'textures': [], 'principled': None}
        if material.use_nodes and material.node_tree:
            for node in material.node_tree.nodes:
                if node.type == 'TEX_IMAGE' and node.image:
                    links = [{'output': link.from_socket.name, 'toNode': link.to_node.type, 'toSocket': link.to_socket.name}
                             for link in material.node_tree.links if link.from_node == node]
                    entry['textures'].append({'image': node.image.name, 'file': os.path.realpath(bpy.path.abspath(node.image.filepath)),
                                              'extension': node.extension, 'interpolation': node.interpolation, 'links': links})
                if node.type == 'BSDF_PRINCIPLED':
                    inputs = {}
                    for socket in node.inputs:
                        if socket.name in ('Base Color', 'Alpha', 'Roughness', 'Metallic', 'Emission Color', 'Emission Strength',
                                           'Specular IOR Level', 'IOR', 'Transmission Weight', 'Normal'):
                            value = socket.default_value if hasattr(socket, 'default_value') else None
                            inputs[socket.name] = {'linked': socket.is_linked,
                                                   'value': rounded(value) if hasattr(value, '__len__') else (round(float(value), 6) if value is not None else None)}
                    entry['principled'] = inputs
        materials[material.name] = entry

    images = {image.name: {'file': os.path.realpath(bpy.path.abspath(image.filepath)) if image.filepath else None,
                           'size': list(image.size), 'colorspace': image.colorspace_settings.name} for image in bpy.data.images}

    lights = []
    for obj in scene.objects:
        if obj.type == 'LIGHT':
            light = obj.data
            lights.append({'name': obj.name, 'type': light.type, 'color': rounded(light.color), 'energy': round(float(light.energy), 6),
                           'position': rounded(to_gltf(obj.matrix_world.translation)),
                           'direction': rounded(to_gltf((obj.matrix_world.to_3x3() @ __import__('mathutils').Vector((0, 0, -1))).normalized())),
                           'spotSize': round(float(getattr(light, 'spot_size', 0.0)), 6), 'spotBlend': round(float(getattr(light, 'spot_blend', 0.0)), 6)})

    cameras = []
    fps = scene.render.fps / scene.render.fps_base
    for obj in scene.objects:
        if obj.type != 'CAMERA':
            continue
        action = obj.animation_data.action if obj.animation_data else None
        frame_range = [int(action.frame_range[0]), int(action.frame_range[1])] if action else [scene.frame_current, scene.frame_current]
        count = frame_range[1] - frame_range[0] + 1
        step = max(1, math.ceil(count / camera_samples)) if camera_samples else count + 1
        path = []
        if camera_samples:
            for frame in range(frame_range[0], frame_range[1] + 1, step):
                scene.frame_set(frame)
                world = obj.matrix_world
                forward = world.to_3x3() @ __import__('mathutils').Vector((0, 0, -1))
                up = world.to_3x3() @ __import__('mathutils').Vector((0, 1, 0))
                path.append({'frame': frame, 'time': round((frame - frame_range[0]) / fps, 6), 'position': rounded(to_gltf(world.translation)),
                             'forward': rounded(to_gltf(forward.normalized())), 'up': rounded(to_gltf(up.normalized()))})
        cameras.append({'name': obj.name, 'animated': action is not None, 'frameRange': frame_range, 'fps': fps, 'sampleStep': step,
                        'verticalFovRadians': round(float(obj.data.angle_y), 6), 'sensorFit': obj.data.sensor_fit,
                        'clip': [round(float(obj.data.clip_start), 6), round(float(obj.data.clip_end), 6)], 'path': path})

    wanted = {'export_format': 'GLTF_SEPARATE', 'export_image_format': 'NONE', 'export_materials': 'EXPORT',
              'export_texcoords': True, 'export_normals': True, 'export_tangents': True, 'export_vertex_color': 'NONE',
              'export_attributes': False, 'export_cameras': False, 'export_lights': True, 'export_animations': False,
              'export_skins': False, 'export_morph': False, 'export_yup': True, 'export_apply': False,
              'export_extras': False, 'export_draco_mesh_compression_enable': False, 'export_gpu_instances': False,
              'use_selection': False, 'use_visible': False, 'use_renderable': False, 'will_save_settings': False}
    available = set(bpy.ops.export_scene.gltf.get_rna_type().properties.keys())
    options = {key: value for key, value in wanted.items() if key in available}
    missing = sorted(key for key in wanted if key not in available)
    if 'export_format' not in options or 'export_image_format' not in options:
        fail('EXPORTER_OPTIONS: ' + ','.join(missing))
    os.makedirs(output)
    gltf_path = os.path.join(output, 'scene.gltf')
    started_export = time.time()
    result = bpy.ops.export_scene.gltf(filepath=gltf_path, **options)
    if 'FINISHED' not in result or not os.path.isfile(gltf_path):
        fail('EXPORT_FAILED')
    export_seconds = time.time() - started_export

    report = {'version': 1, 'format': 'real-engine-blender-fbx-report', 'blender': bpy.app.version_string,
              'source': {'file': os.path.basename(source), 'bytes': os.path.getsize(source), 'sha256': sha256_file(source)},
              'importer': {'operator': 'import_scene.fbx', 'use_custom_normals': True, 'use_image_search': False, 'use_anim': True},
              'exporter': {'operator': 'export_scene.gltf', 'options': options, 'unavailableOptions': missing},
              'units': {'system': scene.unit_settings.system, 'scaleLength': scene.unit_settings.scale_length},
              'boundsBlender': {'min': rounded(lo), 'max': rounded(hi)},
              'boundsGltf': {'min': rounded([lo[0], lo[2], -hi[1]]), 'max': rounded([hi[0], hi[2], -lo[1]])},
              'objects': types, 'meshCount': len(meshes), 'uniqueTriangles': unique_triangles,
              'instanceExpandedTriangles': expanded_triangles, 'sharedMeshes': sum(1 for v in users.values() if v > 1),
              'largestMeshes': [{'name': name, 'triangles': t} for t, name in largest[:16]],
              'materials': materials, 'images': images, 'lights': lights, 'cameras': cameras, 'meshes': meshes,
              'seconds': {'import': round(import_seconds, 3), 'export': round(export_seconds, 3)}}
    with open(os.path.join(output, 'blender-report.json'), 'w') as handle:
        json.dump(report, handle, indent=1)
    print('CONVERT_FBX_DONE ' + json.dumps({'output': output, 'objects': types, 'meshes': len(meshes), 'uniqueTriangles': unique_triangles,
                                             'instanceExpandedTriangles': expanded_triangles, 'materials': len(materials),
                                             'images': len(images), 'lights': len(lights), 'cameras': len(cameras),
                                             'unavailableOptions': missing, 'seconds': report['seconds']}))


main()
