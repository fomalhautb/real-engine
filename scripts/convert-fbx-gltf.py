# Converts an FBX character to glTF 2.0 (separate .gltf + .bin + images) with skins, every action as
# its own glTF animation, and morph targets when the meshes have shape keys. Y-up, no Draco, no lights,
# no cameras. Writes conversion-report.json next to the glTF: counts, checks and everything that was
# dropped or rebuilt. The report holds no absolute paths and no timings; progress and timings go to stdout.
#
#   blender --background --factory-startup --disable-autoexec --threads 1 --python scripts/convert-fbx-gltf.py -- \
#     --input mesh.fbx --output new-directory [--name NAME] [--uv-set map1] [--max-influences 4] \
#     [--animation skeleton.fbx --armature ROOT --skin-ma scene.ma --skin MESH=CLUSTER ...] \
#     [--materials-ma scene.ma --textures directory]
#
# --threads 1 makes the output byte-identical across runs: the exporter's multithreaded MikkTSpace
# tangents can flip a 1e-4 rounding step of a few TANGENT components between runs. Without it the
# output is equivalent but its .bin hash can differ.
#
# Without the optional groups the meshes, armatures and actions of --input are exported as imported.
#
# Skin rebuild (--skin-ma): some Maya FBX exports keep the skin deformers and bind poses but drop the
# cluster weights, so the meshes import without vertex groups. The rebuild reads the weights and bind
# matrices of each named skinCluster from a Maya ASCII scene, checks that the cluster's input shape has
# the same points as the FBX mesh, takes the deforming skeleton and its baked per-frame animation from
# --animation (object --armature is the root joint), rebuilds the rest pose from the inverse
# bindPreMatrix values, re-keys every joint from the sampled world matrices, keeps the largest
# --max-influences weights per vertex, and verifies the Blender deformation against linear blend
# skinning of the Maya data. All other objects of both FBX files (rig controls) are discarded.
#
# Material rebuild (--materials-ma): every FBX material named like a usdPreviewSurface node of the Maya
# scene is rebuilt as a Principled BSDF from that node's values and file textures (found by file name in
# --textures). IOR is pinned to 1.5 and specular to its default so the exporter writes no material
# extensions; the source values are recorded.
#
# Inputs are untrusted data. FBX files are parsed only by Blender's importer, Maya ASCII files only as
# text by this script (their script nodes are never run), and textures are only loaded as images.
# Run Blender with --factory-startup and --disable-autoexec. The output directory must not exist.
import argparse
import json
import math
import os
import re
import sys
import time
import traceback

import bpy
import numpy as np
from mathutils import Matrix

REPORT_NAME = 'conversion-report.json'
REPORT_FORMAT = 'real-engine-fbx-gltf-conversion'
LIMITS = {
    'inputBytes': 1 << 30,
    'mayaAsciiBytes': 256 << 20,
    'meshVertices': 4_000_000,
    'joints': 1024,
    'frames': 20_000,
    'sourceInfluencesPerVertex': 64,
    'images': 64,
    'imagePixels': 8192 * 8192,
}
# Maya scenes are Y-up centimetres. Blender's FBX importer maps them to Z-up metres with this matrix
# (rotate +90 degrees about X, scale 0.01); the skin rebuild checks it against the imported meshes.
MAYA_TO_BLENDER = np.array([[0.01, 0.0, 0.0, 0.0], [0.0, 0.0, -0.01, 0.0], [0.0, 0.01, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
# Converts joint frames with centimetre-sized axes back to unit axes.
CENTIMETRES_PER_METRE = np.diag([100.0, 100.0, 100.0, 1.0])
# The glTF exporter maps Blender Z-up world coordinates to Y-up, (x, y, z) -> (x, z, -y), and keeps each
# bone's own axes: glTF joint world = BLENDER_TO_GLTF @ Blender bone matrix.
BLENDER_TO_GLTF = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
ALLOWED_GLTF_ATTRIBUTES = {'POSITION', 'NORMAL', 'TANGENT', 'TEXCOORD_0', 'JOINTS_0', 'WEIGHTS_0'}
TOKEN = re.compile(r'"(?:[^"\\]|\\.)*"|;|[^\s";]+')
# setAttr flags and the number of arguments each one takes.
SETATTR_FLAGS = {'-av': 0, '-alteredValue': 0, '-c': 0, '-clamp': 0, '-ca': 1, '-caching': 1, '-ch': 1,
                 '-capacityHint': 1, '-cb': 1, '-channelBox': 1, '-k': 1, '-keyable': 1, '-l': 1, '-lock': 1,
                 '-s': 1, '-size': 1, '-type': 1}
# usdPreviewSurface defaults (UsdPreviewSurface specification) for attributes a Maya file leaves unset.
PREVIEW_DEFAULTS = {'dc': (0.18, 0.18, 0.18), 'mtl': 0.0, 'rgh': 0.5, 'ior': 1.5}
PREVIEW_HANDLED = {'dc', 'mtl', 'rgh', 'ior', 'nrm'}
FILE_HANDLED = {'ftn', 'cs', 'cg', 'co', 'ail'}
STARTED = time.perf_counter()


class ConversionError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def fail(code, message):
    raise ConversionError(code, message)


def log(message):
    print(f'[convert-fbx-gltf {time.perf_counter() - STARTED:8.2f}s] {message}', flush=True)


def rounded(value, digits=9):
    if isinstance(value, float):
        return float(f'{value:.{digits}g}')
    if isinstance(value, (list, tuple)):
        return [rounded(item, digits) for item in value]
    if isinstance(value, dict):
        return {key: rounded(item, digits) for key, item in value.items()}
    return value


def short_name(name):
    # Last DAG path component without Maya or FBX namespaces: '|root|knight_rig:hand_l' -> 'hand_l'.
    return name.split('|')[-1].split(':')[-1]


# ---------------------------------------------------------------------------------------------------
# Arguments and inputs
# ---------------------------------------------------------------------------------------------------

def parse_arguments():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    parser = argparse.ArgumentParser(prog='convert-fbx-gltf.py')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--name')
    parser.add_argument('--uv-set', default='map1')
    parser.add_argument('--max-influences', type=int, default=4)
    parser.add_argument('--animation')
    parser.add_argument('--armature')
    parser.add_argument('--skin-ma')
    parser.add_argument('--skin', action='append', default=[])
    parser.add_argument('--materials-ma')
    parser.add_argument('--textures')
    args = parser.parse_args(argv)
    args.input = os.path.realpath(args.input)
    args.output = os.path.abspath(args.output)
    name = args.name or os.path.splitext(os.path.basename(args.input))[0]
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}', name):
        fail('ARGUMENTS', f'--name must be a plain file stem, got {name!r}')
    args.name = name
    if not 1 <= args.max_influences <= 4:
        fail('ARGUMENTS', '--max-influences must be 1..4 (glTF JOINTS_0/WEIGHTS_0 hold four)')
    skin_given = bool(args.skin_ma or args.skin or args.armature or args.animation)
    if skin_given and not (args.skin_ma and args.skin and args.armature):
        fail('ARGUMENTS', '--skin-ma, --skin and --armature are required together (--animation defaults to --input)')
    pairs = []
    for item in args.skin:
        mesh, separator, cluster = item.partition('=')
        if not separator or not mesh or not cluster:
            fail('ARGUMENTS', f'--skin expects MESH=CLUSTER, got {item!r}')
        pairs.append((mesh, cluster))
    if len({mesh for mesh, _ in pairs}) != len(pairs):
        fail('ARGUMENTS', 'each mesh may appear in one --skin pair only')
    args.skin = pairs
    if bool(args.materials_ma) != bool(args.textures):
        fail('ARGUMENTS', '--materials-ma and --textures are required together')
    for key in ('input', 'animation', 'skin_ma', 'materials_ma'):
        value = getattr(args, key)
        if value is None:
            continue
        value = os.path.realpath(value)
        setattr(args, key, value)
        if not os.path.isfile(value):
            fail('INPUT', f'--{key.replace("_", "-")} is not a file: {value}')
        limit = LIMITS['mayaAsciiBytes'] if key.endswith('_ma') else LIMITS['inputBytes']
        if os.path.getsize(value) > limit:
            fail('INPUT_SIZE', f'--{key.replace("_", "-")} exceeds {limit} bytes')
    if args.textures:
        args.textures = os.path.realpath(args.textures)
        if not os.path.isdir(args.textures):
            fail('INPUT', f'--textures is not a directory: {args.textures}')
    if os.path.lexists(args.output):
        fail('OUTPUT_EXISTS', f'output already exists: {args.output}')
    if not os.path.isdir(os.path.dirname(args.output)):
        fail('OUTPUT_PARENT', f'output parent directory does not exist: {os.path.dirname(args.output)}')
    return args


# ---------------------------------------------------------------------------------------------------
# Maya ASCII reader (text only; bounded by the file size limit)
# ---------------------------------------------------------------------------------------------------

def unquote(token):
    if len(token) >= 2 and token[0] == '"' and token[-1] == '"':
        return token[1:-1].replace('\\"', '"').replace('\\\\', '\\')
    return token


def split_statements(text):
    current = []
    for match in TOKEN.finditer(text):
        token = match.group(0)
        if token == ';':
            if current:
                yield current
            current = []
        else:
            current.append(token)
    if current:
        yield current


def parse_setattr(statement):
    attribute = None
    values = []
    i = 1
    while i < len(statement):
        token = statement[i]
        if token in SETATTR_FLAGS:
            i += 1 + SETATTR_FLAGS[token]
            continue
        if attribute is None and token.startswith('"'):
            attribute = unquote(token)
        else:
            values.append(token)
        i += 1
    return attribute, values


class MayaAscii:
    """Node table, connections and per-node setAttr blocks of a Maya ASCII scene."""

    def __init__(self, path):
        with open(path, 'r', encoding='latin-1') as handle:  # Maya ASCII is 8-bit text; latin-1 maps every byte.
            self.text = handle.read()
        self.nodes = {}
        self.connections = []
        text = self.text
        # Top-level commands start at column 0; node attribute statements are indented.
        starts = [match.start() for match in re.finditer(r'^[A-Za-z]', text, re.M)]
        starts.append(len(text))
        for index in range(len(starts) - 1):
            begin, limit = starts[index], starts[index + 1]
            if text.startswith('createNode ', begin):
                end = text.find(';', begin, limit)
                if end < 0:
                    continue
                tokens = TOKEN.findall(text[begin:end])
                node_type = tokens[1] if len(tokens) > 1 else ''
                name = parent = None
                for i, token in enumerate(tokens[:-1]):
                    if token == '-n':
                        name = unquote(tokens[i + 1])
                    elif token == '-p':
                        parent = unquote(tokens[i + 1])
                if name:
                    self.nodes.setdefault(name, []).append({'type': node_type, 'parent': parent, 'start': end + 1, 'end': limit})
            elif text.startswith('connectAttr ', begin):
                end = text.find(';', begin, limit)
                quoted = [unquote(token) for token in TOKEN.findall(text[begin:end if end >= 0 else limit]) if token.startswith('"')]
                if len(quoted) >= 2:
                    self.connections.append((quoted[0], quoted[1]))

    def node(self, reference, node_type=None):
        name = reference.split('|')[-1]
        entries = self.nodes.get(name)
        if not entries:
            fail('MAYA_NODE', f'node {reference!r} not found')
        if len(entries) > 1:
            parent = reference.split('|')[-2] if '|' in reference else None
            entries = [entry for entry in entries if parent and entry['parent'] and entry['parent'].split('|')[-1] == parent]
            if len(entries) != 1:
                fail('MAYA_NODE', f'node name {reference!r} is ambiguous')
        entry = entries[0]
        if node_type and entry['type'] != node_type:
            fail('MAYA_NODE', f'node {reference!r} is a {entry["type"]}, expected {node_type}')
        return entry

    def node_type(self, reference):
        entries = self.nodes.get(reference.split('|')[-1])
        return entries[0]['type'] if entries and len(entries) == 1 else None

    def attributes(self, reference, node_type=None):
        entry = self.node(reference, node_type)
        result = []
        for statement in split_statements(self.text[entry['start']:entry['end']]):
            if statement[0] == 'setAttr':
                attribute, values = parse_setattr(statement)
                if attribute is not None:
                    result.append((attribute, values))
        return result

    def incoming(self, reference):
        name = reference.split('|')[-1]
        return [(source, destination.partition('.')[2]) for source, destination in self.connections
                if destination.partition('.')[0].split('|')[-1] == name]


def matrix_from_maya(values):
    # Maya writes 4x4 matrices row by row for row vectors; transpose for column vectors.
    if len(values) != 16:
        fail('MAYA_MATRIX', f'expected 16 matrix values, got {len(values)}')
    return np.array([float(value) for value in values], dtype=np.float64).reshape(4, 4).T


def read_skin_cluster(maya, name):
    weights = {}
    bind_pre = {}
    settings = {}
    wl_range = re.compile(r'^\.wl\[(\d+)(?::(\d+))?\]\.w$')
    wl_single = re.compile(r'^\.wl\[(\d+)\]\.w\[(\d+)\]$')
    pm_single = re.compile(r'^\.pm\[(\d+)\]$')
    for attribute, values in maya.attributes(name, 'skinCluster'):
        match = wl_range.match(attribute)
        if match:
            low = int(match.group(1))
            high = int(match.group(2) or match.group(1))
            position = 0
            for vertex in range(low, high + 1):
                count = int(values[position])
                position += 1
                if count > LIMITS['sourceInfluencesPerVertex']:
                    fail('SKIN_INFLUENCES', f'{name} vertex {vertex} has {count} influences')
                entry = weights.setdefault(vertex, {})
                for _ in range(count):
                    entry[int(values[position])] = float(values[position + 1])
                    position += 2
            if position != len(values):
                fail('SKIN_WEIGHTS', f'{name} {attribute}: {len(values) - position} unparsed values')
            continue
        match = wl_single.match(attribute)
        if match:
            weights.setdefault(int(match.group(1)), {})[int(match.group(2))] = float(values[0])
            continue
        match = pm_single.match(attribute)
        if match:
            bind_pre[int(match.group(1))] = matrix_from_maya(values)
            continue
        if attribute in ('.gm', '.skm', '.mi', '.nw', '.ucm', '.dqsi', '.dqss', '.bm', '.dpf'):
            settings[attribute[1:]] = values
    influences = {}
    input_shape = None
    for source, plug in maya.incoming(name):
        match = re.fullmatch(r'(?:ma|matrix)\[(\d+)\]', plug)
        if match and re.search(r'\.(wm|worldMatrix)(\[\d+\])?$', source):
            influences[int(match.group(1))] = short_name(source.rpartition('.')[0])
        elif plug in ('ip[0].ig', 'input[0].inputGeometry'):
            input_shape = source.rpartition('.')[0]
    if not weights or not influences or input_shape is None:
        fail('SKIN_CLUSTER', f'{name}: weights, matrix inputs or input geometry missing')
    if sorted(bind_pre) != sorted(influences):
        fail('SKIN_CLUSTER', f'{name}: bindPreMatrix and matrix inputs cover different indices')
    if len(influences) > LIMITS['joints']:
        fail('SKIN_CLUSTER', f'{name}: {len(influences)} influences exceed {LIMITS["joints"]}')
    geometry = matrix_from_maya(settings['gm']) if 'gm' in settings else np.identity(4)
    if np.max(np.abs(geometry - np.identity(4))) > 1e-9:
        fail('SKIN_GEOMETRY_MATRIX', f'{name}: geomMatrix is not identity')
    method = int(settings['skm'][0]) if 'skm' in settings else 0
    return {'name': name, 'weights': weights, 'bindPre': bind_pre, 'influences': influences,
            'inputShape': input_shape, 'skinningMethod': method,
            'settings': {key: ' '.join(values) for key, values in sorted(settings.items()) if key != 'gm'}}


def read_mesh_points(maya, shape):
    points = {}
    tweaks = {}
    pattern = re.compile(r'^\.(vt|pt)\[(\d+)(?::(\d+))?\]$')
    component = re.compile(r'^\.pt\[(\d+)\]\.(px|py|pz)$')
    for attribute, values in maya.attributes(shape, 'mesh'):
        match = pattern.match(attribute)
        if match:
            target = points if match.group(1) == 'vt' else tweaks
            low = int(match.group(2))
            high = int(match.group(3) or match.group(2))
            numbers = [float(value) for value in values]
            if len(numbers) != (high - low + 1) * 3:
                fail('MAYA_POINTS', f'{shape} {attribute}: {len(numbers)} values')
            for k in range(high - low + 1):
                target[low + k] = numbers[k * 3:k * 3 + 3]
            continue
        match = component.match(attribute)
        if match:
            tweaks.setdefault(int(match.group(1)), [0.0, 0.0, 0.0])['xyz'.index(match.group(2)[1])] = float(values[0])
    count = len(points)
    if count == 0 or sorted(points) != list(range(count)):
        fail('MAYA_POINTS', f'{shape}: vertex positions are missing or sparse')
    if count > LIMITS['meshVertices']:
        fail('MAYA_POINTS', f'{shape}: {count} vertices exceed the limit')
    result = np.array([points[i] for i in range(count)], dtype=np.float64)
    for index, value in tweaks.items():
        if index < count:
            result[index] += value
    return result, sum(1 for value in tweaks.values() if any(abs(c) > 0 for c in value))


def read_preview_surfaces(maya, names, textures):
    specs = {}
    for name in names:
        entries = maya.nodes.get(name)
        if not entries or entries[0]['type'] != 'usdPreviewSurface':
            continue
        values = {}
        ignored = []
        for attribute, tokens in maya.attributes(name, 'usdPreviewSurface'):
            key = attribute.lstrip('.')
            if key in ('dc',):
                values[key] = tuple(float(token) for token in tokens[-3:])
            elif key in ('mtl', 'rgh', 'ior'):
                values[key] = float(tokens[-1])
            else:
                ignored.append(key)
        inputs = {}
        for source, plug in maya.incoming(name):
            source_node, _, source_plug = source.rpartition('.')[0], None, source.rpartition('.')[2]
            if maya.node_type(source_node) != 'file':
                ignored.append(f'{plug}<-{maya.node_type(source_node)}')
                continue
            if plug not in PREVIEW_HANDLED:
                ignored.append(f'{plug}<-file')
                continue
            inputs[plug] = read_file_texture(maya, source_node, source_plug, textures)
        specs[name] = {'values': values, 'inputs': inputs, 'ignored': sorted(set(ignored))}
    return specs


def read_file_texture(maya, node, plug, textures):
    values = {}
    ignored = []
    for attribute, tokens in maya.attributes(node, 'file'):
        key = attribute.lstrip('.')
        if key in ('ftn', 'cs'):
            values[key] = unquote(tokens[-1])
        elif key in ('cg', 'co'):
            values[key] = tuple(float(token) for token in tokens[-3:])
        elif key == 'ail':
            values[key] = tokens[-1]
        else:
            ignored.append(key)
    if 'ftn' not in values:
        fail('MAYA_TEXTURE', f'file node {node} has no file name')
    file_name = re.split(r'[\\/]', values['ftn'])[-1]
    path = os.path.join(textures, file_name)
    if not os.path.isfile(path):
        fail('MAYA_TEXTURE', f'texture {file_name} of {node} not found in --textures')
    placement = []
    for source, target_plug in maya.incoming(node):
        source_node = source.rpartition('.')[0]
        source_type = maya.node_type(source_node)
        if source_type == 'place2dTexture':
            placement.append(source_node)
        elif source_type not in ('colorManagementGlobals', None):
            ignored.append(f'{target_plug}<-{source_type}')
    for source_node in sorted(set(placement)):
        if maya.attributes(source_node, 'place2dTexture'):
            fail('MAYA_TEXTURE', f'{source_node} changes the texture placement; UV transforms are not supported')
        for source, _ in maya.incoming(source_node):
            if maya.node_type(source.rpartition('.')[0]) == 'uvChooser':
                fail('MAYA_TEXTURE', f'{source_node} uses a uvChooser; only the default UV set is supported')
    channel = {'oc': 'rgb', 'ocr': 'r', 'ocg': 'g', 'ocb': 'b', 'oa': 'a'}.get(plug)
    if channel is None:
        fail('MAYA_TEXTURE', f'unsupported file output {node}.{plug}')
    return {'node': short_name(node), 'file': file_name, 'path': path, 'colorSpace': values.get('cs', 'sRGB'),
            'gain': values.get('cg', (1.0, 1.0, 1.0)), 'offset': values.get('co', (0.0, 0.0, 0.0)),
            'channel': channel, 'ignored': sorted(set(ignored))}


# ---------------------------------------------------------------------------------------------------
# Blender helpers
# ---------------------------------------------------------------------------------------------------

def import_fbx(path, use_anim):
    wanted = {
        'use_custom_normals': True, 'use_image_search': False, 'use_anim': use_anim, 'anim_offset': 0.0,
        'use_subsurf': False, 'use_custom_props': False, 'ignore_leaf_bones': False, 'force_connect_children': False,
        'automatic_bone_orientation': False, 'primary_bone_axis': 'Y', 'secondary_bone_axis': 'X',
        'use_prepost_rot': True, 'global_scale': 1.0, 'apply_unit_scale': True, 'bake_space_transform': False,
        'use_manual_orientation': False,
    }
    available = set(bpy.ops.import_scene.fbx.get_rna_type().properties.keys())
    applied = {key: value for key, value in wanted.items() if key in available}
    before = set(bpy.data.objects)
    result = bpy.ops.import_scene.fbx(filepath=path, **applied)
    if 'FINISHED' not in result:
        fail('FBX_IMPORT', f'FBX import failed: {os.path.basename(path)}')
    created = [obj for obj in bpy.data.objects if obj not in before]
    return created, applied, sorted(set(wanted) - set(applied))


def type_counts(objects):
    counts = {}
    for obj in objects:
        counts[obj.type] = counts.get(obj.type, 0) + 1
    return dict(sorted(counts.items()))


def remove_objects(objects):
    for obj in list(objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def purge_orphans():
    for collection in (bpy.data.meshes, bpy.data.armatures, bpy.data.curves, bpy.data.cameras, bpy.data.lights,
                       bpy.data.materials, bpy.data.images, bpy.data.actions):
        for block in list(collection):
            if block.users == 0:
                collection.remove(block)


def channelbag_of(id_block):
    from bpy_extras import anim_utils
    data = id_block.animation_data
    if data is None or data.action is None:
        return None
    return anim_utils.action_get_channelbag_for_slot(data.action, data.action_slot)


def numpy_matrix(matrix):
    return np.array([list(row) for row in matrix], dtype=np.float64)


def blender_matrix(array):
    return Matrix([list(map(float, row)) for row in array])


def rotation_error_radians(a, b):
    # Angle of the relative rotation between the orthonormalised 3x3 parts.
    ra = a[:3, :3] / np.linalg.norm(a[:3, :3], axis=0)
    rb = b[:3, :3] / np.linalg.norm(b[:3, :3], axis=0)
    cosine = (np.trace(ra.T @ rb) - 1.0) * 0.5
    return math.acos(max(-1.0, min(1.0, cosine)))


def mesh_positions(mesh):
    values = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
    mesh.vertices.foreach_get('co', values)
    return values.reshape(-1, 3).astype(np.float64)


def corner_normals(mesh):
    values = np.empty(len(mesh.loops) * 3, dtype=np.float32)
    mesh.corner_normals.foreach_get('vector', values)
    return values.reshape(-1, 3).astype(np.float64)


def bake_transform_and_triangulate(obj, matrix):
    # Bakes `matrix` into the vertices and splits polygons with more than four corners into triangles
    # (glTF stores triangles, and Blender computes MikkTSpace tangents only for triangles and quads).
    # The corner normals are carried through both steps in a corner attribute and written back as
    # custom normals, so neither step changes the shading normals beyond custom-normal encoding precision.
    import bmesh
    mesh = obj.data
    rotation = numpy_matrix(matrix)[:3, :3]
    rotation = rotation / np.linalg.norm(rotation, axis=0)
    target = corner_normals(mesh) @ rotation.T
    attribute = mesh.attributes.new('_conversion_normal', 'FLOAT_VECTOR', 'CORNER')
    attribute.data.foreach_set('vector', target.astype(np.float32).ravel())
    if np.max(np.abs(numpy_matrix(matrix) - np.identity(4))) > 0:
        mesh.transform(matrix)
    ngons = sum(1 for polygon in mesh.polygons if polygon.loop_total > 4)
    if ngons:
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bmesh.ops.triangulate(bm, faces=[face for face in bm.faces if len(face.verts) > 4],
                              quad_method='BEAUTY', ngon_method='BEAUTY')
        bm.to_mesh(mesh)
        bm.free()
    values = np.empty(len(mesh.loops) * 3, dtype=np.float32)
    mesh.attributes['_conversion_normal'].data.foreach_get('vector', values)
    mesh.attributes.remove(mesh.attributes['_conversion_normal'])
    values = values.reshape(-1, 3).astype(np.float64)
    values /= np.maximum(np.linalg.norm(values, axis=1), 1e-12)[:, None]
    mesh.normals_split_custom_set(values.tolist())
    after = corner_normals(mesh)
    after /= np.maximum(np.linalg.norm(after, axis=1), 1e-12)[:, None]
    angle = float(np.max(np.arccos(np.clip(np.sum(after * values, axis=1), -1.0, 1.0))))
    return ngons, angle


def evaluated_positions(obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        return mesh_positions(mesh)
    finally:
        evaluated.to_mesh_clear()


# ---------------------------------------------------------------------------------------------------
# Skinning math (column vectors, float64)
# ---------------------------------------------------------------------------------------------------

def blend_linear(skin, positions, influences):
    vertex, joint, weight = influences
    matrices = skin[joint]
    moved = np.einsum('nij,nj->ni', matrices[:, :3, :3], positions[vertex]) + matrices[:, :3, 3]
    result = np.zeros_like(positions)
    np.add.at(result, vertex, moved * weight[:, None])
    return result


def rotation_to_quaternion(rotation):
    # Shepperd's method on an orthonormal 3x3 matrix; returns (w, x, y, z).
    m = rotation
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        q = (0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s)
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = ((m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s)
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = ((m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s)
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = ((m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s)
    q = np.array(q)
    return q / np.linalg.norm(q)


def blend_dual_quaternion(skin, positions, influences, first_joint):
    # Kavan-style dual quaternion skinning with antipodality resolved against each vertex's largest influence.
    vertex, joint, weight = influences
    real = np.array([rotation_to_quaternion(m[:3, :3] / np.linalg.norm(m[:3, :3], axis=0)) for m in skin])
    t = skin[:, :3, 3]
    dual_w = -0.5 * np.sum(t * real[:, 1:], axis=1)
    dual_v = 0.5 * (real[:, :1] * t + np.cross(t, real[:, 1:]))
    dual = np.concatenate([dual_w[:, None], dual_v], axis=1)
    sign = np.sign(np.sum(real[joint] * real[first_joint[vertex]], axis=1))
    sign[sign == 0] = 1.0
    blend_real = np.zeros((len(positions), 4))
    blend_dual = np.zeros((len(positions), 4))
    np.add.at(blend_real, vertex, real[joint] * (weight * sign)[:, None])
    np.add.at(blend_dual, vertex, dual[joint] * (weight * sign)[:, None])
    norm = np.linalg.norm(blend_real, axis=1)[:, None]
    r = blend_real / norm
    d = blend_dual / norm
    rw, rv = r[:, :1], r[:, 1:]
    dw, dv = d[:, :1], d[:, 1:]
    p = positions
    rotated = p + 2.0 * rw * np.cross(rv, p) + 2.0 * np.cross(rv, np.cross(rv, p))
    translation = 2.0 * (rw * dv - dw * rv + np.cross(rv, dv))
    return rotated + translation


# ---------------------------------------------------------------------------------------------------
# Skin rebuild
# ---------------------------------------------------------------------------------------------------

def rebuild_skin(args, scene, meshes_by_name, report):
    maya_started = time.perf_counter()
    maya = MayaAscii(args.skin_ma)
    log(f'skin scene parsed: {sum(len(v) for v in maya.nodes.values())} nodes, {len(maya.connections)} connections '
        f'({time.perf_counter() - maya_started:.2f}s)')
    clusters = []
    for mesh_name, cluster_name in args.skin:
        if mesh_name not in meshes_by_name:
            fail('SKIN_MESH', f'--skin mesh {mesh_name!r} is not a mesh object of --input')
        cluster = read_skin_cluster(maya, cluster_name)
        points, tweaked = read_mesh_points(maya, cluster['inputShape'])
        cluster.update({'mesh': mesh_name, 'origPoints': points, 'tweakedPoints': tweaked})
        clusters.append(cluster)
    maya = None

    # Joint set and bind matrices (Maya world, centimetres), shared by every cluster.
    bind = {}
    bind_agreement = 0.0
    for cluster in clusters:
        for index, joint in cluster['influences'].items():
            matrix = np.linalg.inv(cluster['bindPre'][index])
            if joint in bind:
                bind_agreement = max(bind_agreement, float(np.max(np.abs(bind[joint] - matrix))))
            else:
                bind[joint] = matrix
    if bind_agreement > 1e-4:
        fail('SKIN_BIND', f'clusters disagree on bind matrices by {bind_agreement} (Maya units)')
    rigidity = 0.0
    for joint, matrix in bind.items():
        rotation = matrix[:3, :3]
        rigidity = max(rigidity, float(np.max(np.abs(rotation.T @ rotation - np.identity(3)))))
        if np.linalg.det(rotation) <= 0:
            fail('SKIN_BIND', f'bind matrix of {joint} is mirrored')
    if rigidity > 1e-5:
        fail('SKIN_BIND', f'bind matrices carry scale or shear ({rigidity}); Blender rest bones are rigid')

    # Mesh points must equal the cluster input shapes mapped by the importer's Maya-to-Blender matrix.
    mesh_reports = {}
    for cluster in clusters:
        obj = meshes_by_name[cluster['mesh']]
        world = numpy_matrix(obj.matrix_world)
        if np.max(np.abs(world - MAYA_TO_BLENDER)) > 1e-6:
            fail('SKIN_MESH_TRANSFORM', f'{cluster["mesh"]} world matrix is not the Maya-to-Blender conversion')
        positions = mesh_positions(obj.data)
        if len(positions) != len(cluster['origPoints']):
            fail('SKIN_MESH_POINTS', f'{cluster["mesh"]} has {len(positions)} vertices, '
                 f'{cluster["inputShape"]} has {len(cluster["origPoints"])}')
        error = float(np.max(np.linalg.norm(positions - cluster['origPoints'], axis=1)))
        if error > 1e-3:
            fail('SKIN_MESH_POINTS', f'{cluster["mesh"]} differs from {cluster["inputShape"]} by {error} cm')
        missing = [v for v in range(len(positions)) if v not in cluster['weights']]
        if missing:
            fail('SKIN_WEIGHTS', f'{cluster["mesh"]}: {len(missing)} vertices have no weights')
        mesh_reports[cluster['mesh']] = {
            'skinCluster': cluster['name'], 'inputShape': short_name(cluster['inputShape']),
            'pointMaxDifferenceCm': error, 'inputShapeTweakedPoints': cluster['tweakedPoints'],
            'mayaSkinningMethod': {0: 'linear', 1: 'dual-quaternion', 2: 'weight-blended'}.get(cluster['skinningMethod'], str(cluster['skinningMethod'])),
            'mayaSkinClusterSettings': cluster['settings'],
        }

    # Skeleton and baked animation from the animation FBX.
    animation_path = args.animation or args.input
    created, applied, unavailable = import_fbx(animation_path, True)
    report['fbxImport']['animation'] = {'file': os.path.basename(animation_path), 'objects': type_counts(created),
                                        'options': applied, 'unavailableOptions': unavailable}
    armature = next((obj for obj in created if obj.name == args.armature), None)
    if armature is None or armature.type != 'ARMATURE':
        fail('SKELETON', f'--armature {args.armature!r} is not an armature object of the animation FBX')
    source_actions = sorted(action.name for action in bpy.data.actions)
    remove_objects([obj for obj in created if obj is not armature])
    for action in list(bpy.data.actions):
        if armature.animation_data is None or action != armature.animation_data.action:
            bpy.data.actions.remove(action)
    if armature.animation_data is None or armature.animation_data.action is None:
        fail('SKELETON', f'{args.armature} has no action')
    source_action = armature.animation_data.action
    bag = channelbag_of(armature)
    frames_seen = set()
    source_keys = 0
    interpolation = {}
    for curve in bag.fcurves:
        count = len(curve.keyframe_points)
        values = np.empty(count * 2, dtype=np.float32)
        curve.keyframe_points.foreach_get('co', values)
        frames_seen.update(np.round(values[0::2], 4).tolist())
        source_keys += count
        for point in curve.keyframe_points[:1]:
            interpolation[point.interpolation] = interpolation.get(point.interpolation, 0) + 1
    if any(abs(frame - round(frame)) > 1e-3 for frame in frames_seen):
        fail('SKELETON_FRAMES', 'the skeleton action has keys between frames')
    frame_start, frame_end = int(round(min(frames_seen))), int(round(max(frames_seen)))
    frames = list(range(frame_start, frame_end + 1))
    if len(frames) > LIMITS['frames']:
        fail('SKELETON_FRAMES', f'{len(frames)} frames exceed {LIMITS["frames"]}')

    root_name = short_name(armature.name)
    bones = {short_name(bone.name): bone.name for bone in armature.data.bones}
    if len(bones) != len(armature.data.bones) or root_name in bones:
        fail('SKELETON', 'joint names collide after removing namespaces')
    skeleton = {root_name, *bones}
    if set(bind) != skeleton:
        fail('SKELETON', f'skin joints and skeleton differ: only in skin {sorted(set(bind) - skeleton)}, '
             f'only in skeleton {sorted(skeleton - set(bind))}')
    first = clusters[0]['influences']
    joints = [first[index] for index in sorted(first)] + sorted(set(bind) - set(first.values()))
    if joints[0] != root_name:
        joints.remove(root_name)
        joints.insert(0, root_name)
    index_of = {joint: i for i, joint in enumerate(joints)}
    parents = []
    for joint in joints:
        if joint == root_name:
            parents.append(-1)
            continue
        parent = armature.data.bones[bones[joint]].parent
        parents.append(index_of[short_name(parent.name)] if parent else index_of[root_name])
    # Parents before children, so the hierarchy can be walked in order.
    order = []
    placed = set()
    while len(order) < len(joints):
        progressed = False
        for j, parent in enumerate(parents):
            if j not in placed and (parent < 0 or parent in placed):
                order.append(j)
                placed.add(j)
                progressed = True
        if not progressed:
            fail('SKELETON', 'joint hierarchy has a cycle')

    if abs(scene.render.fps_base - 1.0) > 1e-9:
        fail('SKELETON_FRAMES', f'fractional frame rate {scene.render.fps}/{scene.render.fps_base} is not supported')
    fps = scene.render.fps
    sample_started = time.perf_counter()
    worlds = np.empty((len(frames), len(joints), 4, 4))
    lookups = [None if joint == root_name else armature.pose.bones[bones[joint]] for joint in joints]
    for f, frame in enumerate(frames):
        scene.frame_set(frame)
        arm_world = armature.matrix_world.copy()
        for j, pose_bone in enumerate(lookups):
            worlds[f, j] = numpy_matrix(arm_world if pose_bone is None else arm_world @ pose_bone.matrix)
    log(f'sampled {len(joints)} joints over frames {frame_start}..{frame_end} ({time.perf_counter() - sample_started:.2f}s)')
    source_action_name = source_action.name
    take = source_action_name.split('|')[1] if source_action_name.count('|') == 2 else source_action_name
    bpy.data.objects.remove(armature, do_unlink=True)
    bpy.data.actions.remove(source_action)
    purge_orphans()

    # Rest and pose matrices of the new armature (Blender world = armature space, metres, unit axes).
    rest = np.array([MAYA_TO_BLENDER @ bind[joint] @ CENTIMETRES_PER_METRE for joint in joints])
    poses = worlds @ CENTIMETRES_PER_METRE
    axis_lengths = np.linalg.norm(poses[:, :, :3, :3], axis=2)
    joint_scale_range = [float(axis_lengths.min()), float(axis_lengths.max())]

    armature_data = bpy.data.armatures.new(args.name)
    skeleton_object = bpy.data.objects.new(args.name, armature_data)
    scene.collection.objects.link(skeleton_object)
    bpy.context.view_layer.objects.active = skeleton_object
    skeleton_object.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    children = {j: [k for k in range(len(joints)) if parents[k] == j] for j in range(len(joints))}
    lengths = {}
    for j in order:
        head = rest[j][:3, 3]
        distances = [float(np.linalg.norm(rest[k][:3, 3] - head)) for k in children[j]]
        distances = [d for d in distances if d > 1e-4]
        length = min(distances) if distances else (lengths[parents[j]] * 0.5 if parents[j] >= 0 else 0.1)
        lengths[j] = min(max(length, 0.002), 1.0)
        bone = armature_data.edit_bones.new(joints[j])
        bone.head = (0.0, 0.0, 0.0)
        bone.tail = (0.0, lengths[j], 0.0)
        bone.roll = 0.0
        bone.matrix = blender_matrix(rest[j])
        bone.use_connect = False
        bone.use_deform = True
        if parents[j] >= 0:
            bone.parent = armature_data.edit_bones[joints[parents[j]]]
    bpy.ops.object.mode_set(mode='OBJECT')
    rest_actual = np.array([numpy_matrix(armature_data.bones[joint].matrix_local) for joint in joints])
    rest_translation_error = float(np.max(np.linalg.norm(rest_actual[:, :3, 3] - rest[:, :3, 3], axis=1)))
    rest_rotation_errors = [rotation_error_radians(rest_actual[j], rest[j]) for j in range(len(joints))]
    rest_rotation_error = max(rest_rotation_errors)
    rest_outliers = {joints[j]: {'radians': rest_rotation_errors[j], 'boneLengthM': lengths[j],
                                 'restYAxis': [float(v) for v in rest[j][:3, 1]]}
                     for j in range(len(joints)) if rest_rotation_errors[j] > 1e-5}

    # Pose basis per frame: pose = parent_pose @ (parent_rest^-1 @ rest) @ basis (Blender's default inheritance).
    key_started = time.perf_counter()
    inverse_rest = np.linalg.inv(rest_actual)
    basis = np.empty_like(poses)
    for j in range(len(joints)):
        parent = parents[j]
        if parent < 0:
            basis[:, j] = inverse_rest[j] @ poses[:, j]
        else:
            basis[:, j] = inverse_rest[j] @ rest_actual[parent] @ np.linalg.inv(poses[:, parent]) @ poses[:, j]
    action = bpy.data.actions.new(take)
    skeleton_object.animation_data_create()
    slot = action.slots.new(id_type='OBJECT', name=skeleton_object.name)
    skeleton_object.animation_data.action = action
    skeleton_object.animation_data.action_slot = slot
    from bpy_extras import anim_utils
    bag = anim_utils.action_ensure_channelbag_for_slot(action, slot)
    decomposition_error = 0.0
    keyed_curves = 0
    frame_values = np.array(frames, dtype=np.float64)
    for j, joint in enumerate(joints):
        locations = np.empty((len(frames), 3))
        rotations = np.empty((len(frames), 4))
        scales = np.empty((len(frames), 3))
        previous = None
        for f in range(len(frames)):
            matrix = blender_matrix(basis[f, j])
            location, rotation, scale = matrix.decompose()
            quaternion = np.array(rotation)
            if previous is not None and float(np.dot(quaternion, previous)) < 0:
                quaternion = -quaternion
            previous = quaternion
            locations[f], rotations[f], scales[f] = location, quaternion, scale
            recomposed = numpy_matrix(Matrix.LocRotScale(location, rotation, scale))
            decomposition_error = max(decomposition_error, float(np.max(np.abs(recomposed - basis[f, j]))))
        path = f'pose.bones["{bpy.utils.escape_identifier(joint)}"]'
        for prop, data in (('location', locations), ('rotation_quaternion', rotations), ('scale', scales)):
            for axis in range(data.shape[1]):
                curve = bag.fcurves.new(f'{path}.{prop}', index=axis)
                points = curve.keyframe_points
                points.add(len(frames))
                co = np.empty(len(frames) * 2, dtype=np.float64)
                co[0::2] = frame_values
                co[1::2] = data[:, axis]
                points.foreach_set('co', co)
                try:
                    points.foreach_set('interpolation', [1] * len(frames))  # 1 = LINEAR
                except (TypeError, AttributeError):
                    for point in points:
                        point.interpolation = 'LINEAR'
                curve.update()
                keyed_curves += 1
    if bag.fcurves[0].keyframe_points[0].interpolation != 'LINEAR':
        fail('ANIMATION_KEYS', 'keyframe interpolation was not set to LINEAR')
    log(f'keyed {keyed_curves} curves x {len(frames)} frames ({time.perf_counter() - key_started:.2f}s)')
    scene.frame_start, scene.frame_end = frame_start, frame_end

    # The keyed pose must reproduce the sampled joint matrices at every frame.
    pose_translation_error = 0.0
    pose_rotation_error = 0.0
    pose_bones = [skeleton_object.pose.bones[joint] for joint in joints]
    for f, frame in enumerate(frames):
        scene.frame_set(frame)
        for j, pose_bone in enumerate(pose_bones):
            actual = numpy_matrix(pose_bone.matrix)
            pose_translation_error = max(pose_translation_error, float(np.linalg.norm(actual[:3, 3] - poses[f, j][:3, 3])))
            pose_rotation_error = max(pose_rotation_error, rotation_error_radians(actual, poses[f, j]))
    log(f'pose check: {pose_translation_error:.3g} m, {pose_rotation_error:.3g} rad')

    # Meshes: bake the import transform into the vertices, parent to the skeleton, add the weights.
    weight_started = time.perf_counter()
    mesh_data = {}
    for cluster in clusters:
        obj = meshes_by_name[cluster['mesh']]
        world = obj.matrix_world.copy()
        ngons, normal_error = bake_transform_and_triangulate(obj, world)
        obj.matrix_world = Matrix.Identity(4)
        obj.parent = skeleton_object
        obj.matrix_parent_inverse = Matrix.Identity(4)
        for modifier in list(obj.modifiers):
            obj.modifiers.remove(modifier)
        obj.vertex_groups.clear()
        modifier = obj.modifiers.new('Armature', 'ARMATURE')
        modifier.object = skeleton_object
        modifier.use_vertex_groups = True
        modifier.use_bone_envelopes = False
        modifier.use_deform_preserve_volume = False
        cluster_joints = cluster['influences']
        groups = {}
        full = ([], [], [])
        pruned = ([], [], [])
        first_joint = np.empty(len(obj.data.vertices), dtype=np.int64)
        histogram = {}
        dropped = []
        for vertex in range(len(obj.data.vertices)):
            items = sorted(((weight, index_of[cluster_joints[i]]) for i, weight in cluster['weights'][vertex].items() if weight > 0),
                           key=lambda item: (-item[0], item[1]))
            if not items:
                fail('SKIN_WEIGHTS', f'{cluster["mesh"]} vertex {vertex} has no positive weight')
            total = sum(weight for weight, _ in items)
            kept = items[:args.max_influences]
            kept_total = sum(weight for weight, _ in kept)
            histogram[len(items)] = histogram.get(len(items), 0) + 1
            if len(items) > len(kept):
                dropped.append(1.0 - kept_total / total)
            first_joint[vertex] = items[0][1]
            for weight, joint_index in items:
                full[0].append(vertex)
                full[1].append(joint_index)
                full[2].append(weight / total)
            for weight, joint_index in kept:
                pruned[0].append(vertex)
                pruned[1].append(joint_index)
                pruned[2].append(weight / kept_total)
                group = groups.get(joint_index)
                if group is None:
                    group = groups[joint_index] = obj.vertex_groups.new(name=joints[joint_index])
                group.add([vertex], weight / kept_total, 'REPLACE')
        rest_positions = mesh_positions(obj.data)
        mesh_data[cluster['mesh']] = {
            'object': obj, 'positions': rest_positions, 'firstJoint': first_joint,
            'full': tuple(np.array(column) for column in full), 'pruned': tuple(np.array(column) for column in pruned),
        }
        sums = np.zeros(len(rest_positions))
        np.add.at(sums, np.array(full[0]), np.array(full[2]))
        mesh_reports[cluster['mesh']].update({
            'polygonsTriangulated': ngons,
            'customNormalMaxAngleChangeRadians': normal_error,
            'influenceHistogram': {str(key): histogram[key] for key in sorted(histogram)},
            'maxSourceInfluences': max(histogram),
            'keptInfluences': args.max_influences,
            'verticesPruned': len(dropped),
            'droppedWeightMax': max(dropped) if dropped else 0.0,
            'droppedWeightMean': (sum(dropped) / len(dropped)) if dropped else 0.0,
            'jointsUsed': len(groups),
        })
    log(f'weights written ({time.perf_counter() - weight_started:.2f}s)')

    # Deformation check: Blender's evaluated meshes against linear blend skinning of the kept weights,
    # plus the effect of pruning and of linear (glTF) versus dual quaternion (Maya setting) skinning.
    check_frames = sorted({frame_start, min(frame_start + int(fps), frame_end), (frame_start + frame_end) // 2, frame_end})
    rigid_skin = True
    inverse_bind = np.linalg.inv(rest)
    deformation = {name: {'frames': check_frames, 'blenderVsKeptLinearMaxM': 0.0, 'blenderVsMayaBindKeptLinearMaxM': 0.0,
                          'keptVsAllLinearMaxM': 0.0,
                          'keptVsAllLinearMeanM': 0.0, 'linearVsDualQuaternionMaxM': 0.0, 'linearVsDualQuaternionMeanM': 0.0}
                   for name in mesh_data}
    for frame in check_frames:
        scene.frame_set(frame)
        skin = poses[frame - frame_start] @ inverse_rest
        scale_error = float(np.max(np.abs(np.linalg.norm(skin[:, :3, :3], axis=1) - 1.0)))
        rigid_skin = rigid_skin and scale_error < 1e-4
        for name, data in mesh_data.items():
            actual = evaluated_positions(data['object'])
            kept = blend_linear(skin, data['positions'], data['pruned'])
            everything = blend_linear(skin, data['positions'], data['full'])
            dual = blend_dual_quaternion(skin, data['positions'], data['full'], data['firstJoint'])
            entry = deformation[name]
            entry['blenderVsKeptLinearMaxM'] = max(entry['blenderVsKeptLinearMaxM'], float(np.max(np.linalg.norm(actual - kept, axis=1))))
            bind_kept = blend_linear(poses[frame - frame_start] @ inverse_bind, data['positions'], data['pruned'])
            entry['blenderVsMayaBindKeptLinearMaxM'] = max(entry['blenderVsMayaBindKeptLinearMaxM'],
                                                           float(np.max(np.linalg.norm(actual - bind_kept, axis=1))))
            difference = np.linalg.norm(kept - everything, axis=1)
            entry['keptVsAllLinearMaxM'] = max(entry['keptVsAllLinearMaxM'], float(difference.max()))
            entry['keptVsAllLinearMeanM'] = max(entry['keptVsAllLinearMeanM'], float(difference.mean()))
            difference = np.linalg.norm(everything - dual, axis=1)
            entry['linearVsDualQuaternionMaxM'] = max(entry['linearVsDualQuaternionMaxM'], float(difference.max()))
            entry['linearVsDualQuaternionMeanM'] = max(entry['linearVsDualQuaternionMeanM'], float(difference.mean()))
    scene.frame_set(frame_start)
    for name, entry in deformation.items():
        mesh_reports[name]['deformationCheck'] = entry
    log('deformation check: ' + ', '.join(f'{name} {entry["blenderVsKeptLinearMaxM"]:.3g} m' for name, entry in deformation.items()))

    report['skeleton'] = {
        'object': args.name, 'joints': len(joints), 'root': root_name, 'jointNames': joints,
        'parents': [joints[p] if p >= 0 else None for p in parents],
        'restPose': 'inverse skinCluster bindPreMatrix (Maya bind pose), converted to Y-up metres',
        'bindMatrixClusterAgreementMaya': bind_agreement, 'bindMatrixOrthonormalityError': rigidity,
        'restTranslationErrorM': rest_translation_error, 'restRotationErrorRadians': rest_rotation_error,
        'restRotationErrorsAbove1e-5': rest_outliers,
        'jointAxisLengthRange': joint_scale_range, 'skinMatricesRigidAtCheckFrames': rigid_skin,
    }
    report['animation'] = {
        'name': take, 'sourceAction': source_action_name,
        'sourceActionsInFile': len(source_actions), 'sourceActionsDiscarded': len(source_actions) - 1,
        'sourceKeys': source_keys, 'sourceInterpolation': interpolation,
        'frameStart': frame_start, 'frameEnd': frame_end, 'fps': fps, 'frames': len(frames),
        'timeMapping': f'glTF time = (frame - {frame_start}) / {fps}',
        'durationSeconds': (frame_end - frame_start) / fps,
        'keying': 'one LINEAR key per frame for location, rotation_quaternion and scale of every joint',
        'keyedCurves': keyed_curves, 'basisDecompositionMaxError': decomposition_error,
        'poseTranslationMaxErrorM': pose_translation_error, 'poseRotationMaxErrorRadians': pose_rotation_error,
    }
    return mesh_reports, {'joints': joints, 'rest': rest}


# ---------------------------------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------------------------------

def rebuild_materials(args, materials, report):
    maya = MayaAscii(args.materials_ma)
    specs = read_preview_surfaces(maya, [material.name for material in materials], args.textures)
    maya = None
    images = {}
    entries = []
    for material in materials:
        spec = specs.get(material.name)
        if spec is None:
            entries.append({'name': material.name, 'source': None, 'note': 'no usdPreviewSurface of this name; kept as imported'})
            continue
        values = dict(PREVIEW_DEFAULTS)
        values.update(spec['values'])
        tree = material.node_tree
        if tree is None:
            fail('MATERIAL', f'{material.name} has no node tree')
        tree.nodes.clear()
        output = tree.nodes.new('ShaderNodeOutputMaterial')
        shader = tree.nodes.new('ShaderNodeBsdfPrincipled')
        tree.links.new(shader.outputs['BSDF'], output.inputs['Surface'])
        shader.inputs['Base Color'].default_value = (*values['dc'], 1.0)
        shader.inputs['Metallic'].default_value = values['mtl']
        shader.inputs['Roughness'].default_value = values['rgh']
        # Pinned so the exporter writes no KHR_materials_ior / KHR_materials_specular.
        shader.inputs['IOR'].default_value = 1.5
        shader.inputs['Specular IOR Level'].default_value = 0.5
        shader.inputs['Specular Tint'].default_value = (1.0, 1.0, 1.0, 1.0)
        shader.inputs['Emission Strength'].default_value = 0.0
        shader.inputs['Coat Weight'].default_value = 0.0
        shader.inputs['Alpha'].default_value = 1.0
        wiring = {}
        for plug, texture in sorted(spec['inputs'].items()):
            image = images.get(texture['file'])
            if image is None:
                image = bpy.data.images.load(texture['path'], check_existing=False)
                image.name = texture['file']
                width, height = image.size
                if width * height > LIMITS['imagePixels']:
                    fail('TEXTURE_SIZE', f'{texture["file"]} has {width}x{height} pixels')
                images[texture['file']] = image
                if len(images) > LIMITS['images']:
                    fail('TEXTURE_COUNT', 'too many textures')
            image.colorspace_settings.name = 'sRGB' if texture['colorSpace'].lower() == 'srgb' else 'Non-Color'
            node = tree.nodes.new('ShaderNodeTexImage')
            node.image = image
            node.interpolation = 'Linear'
            node.extension = 'REPEAT'
            if plug == 'dc':
                if texture['channel'] != 'rgb' or texture['gain'] != (1.0, 1.0, 1.0) or texture['offset'] != (0.0, 0.0, 0.0):
                    fail('MATERIAL', f'{material.name}: base colour texture with a channel, gain or offset is not supported')
                tree.links.new(node.outputs['Color'], shader.inputs['Base Color'])
                wiring['baseColor'] = texture['file']
            elif plug in ('rgh', 'mtl'):
                if texture['channel'] not in ('r', 'g', 'b') or texture['gain'] != (1.0, 1.0, 1.0) or texture['offset'] != (0.0, 0.0, 0.0):
                    fail('MATERIAL', f'{material.name}: {plug} texture must be one channel without gain or offset')
                split = tree.nodes.new('ShaderNodeSeparateColor')
                split.mode = 'RGB'
                tree.links.new(node.outputs['Color'], split.inputs['Color'])
                socket = {'r': 'Red', 'g': 'Green', 'b': 'Blue'}[texture['channel']]
                target = 'Roughness' if plug == 'rgh' else 'Metallic'
                tree.links.new(split.outputs[socket], shader.inputs[target])
                wiring[target.lower()] = f'{texture["file"]}:{texture["channel"].upper()}'
            elif plug == 'nrm':
                # The Maya file node decodes [0, 1] to [-1, 1] with gain 2 and offset -1, the glTF normal-map encoding.
                if texture['channel'] != 'rgb' or texture['gain'] != (2.0, 2.0, 2.0) or texture['offset'] != (-1.0, -1.0, -1.0):
                    fail('MATERIAL', f'{material.name}: normal texture must use gain 2 and offset -1')
                normal_map = tree.nodes.new('ShaderNodeNormalMap')
                normal_map.space = 'TANGENT'
                normal_map.uv_map = args.uv_set
                tree.links.new(node.outputs['Color'], normal_map.inputs['Color'])
                tree.links.new(normal_map.outputs['Normal'], shader.inputs['Normal'])
                wiring['normal'] = texture['file']
        source = {key: values[key] for key in ('dc', 'mtl', 'rgh', 'ior') if key not in spec['inputs']}
        dropped = {}
        if abs(values['ior'] - 1.5) > 1e-6:
            f0 = ((values['ior'] - 1.0) / (values['ior'] + 1.0)) ** 2
            dropped['ior'] = {'source': values['ior'], 'sourceDielectricF0': f0, 'exported': 1.5, 'exportedDielectricF0': 0.04,
                              'effect': 'none (metallic 1 uses the base colour as F0)' if values['mtl'] >= 1.0 and 'mtl' not in spec['inputs']
                              else 'dielectric specular changes'}
        entries.append({'name': short_name(material.name), 'source': short_name(material.name), 'constants': source,
                        'textures': wiring, 'ignoredSourceAttributes': spec['ignored'],
                        'ignoredTextureAttributes': sorted({item for texture in spec['inputs'].values() for item in texture['ignored']}),
                        'dropped': dropped})
    report['materials'] = entries
    return {name: image for name, image in images.items()}, specs


# ---------------------------------------------------------------------------------------------------
# Export and output checks
# ---------------------------------------------------------------------------------------------------

def export_gltf(args, report):
    filepath = os.path.join(args.output, args.name + '.gltf')
    wanted = {
        'export_format': 'GLTF_SEPARATE', 'export_texture_dir': '', 'export_image_format': 'AUTO',
        'export_keep_originals': False, 'export_unused_images': False, 'export_unused_textures': False,
        'export_texcoords': True, 'export_normals': True, 'export_tangents': True, 'export_materials': 'EXPORT',
        'export_vertex_color': 'NONE', 'export_all_vertex_colors': False, 'export_attributes': False,
        'use_mesh_edges': False, 'use_mesh_vertices': False, 'export_shared_accessors': False,
        'use_selection': False, 'use_visible': False, 'use_renderable': False, 'use_active_collection': False,
        'export_yup': True, 'export_apply': False, 'export_extras': False, 'export_copyright': '',
        'export_animations': True, 'export_animation_mode': 'ACTIONS', 'export_frame_range': False,
        'export_frame_step': 1, 'export_force_sampling': True, 'export_sampling_interpolation_fallback': 'LINEAR',
        'export_anim_slide_to_zero': True, 'export_negative_frame': 'SLIDE', 'export_bake_animation': False,
        'export_current_frame': False, 'export_optimize_animation_size': True,
        'export_optimize_animation_keep_anim_armature': True, 'export_optimize_animation_keep_anim_object': False,
        'export_anim_single_armature': True, 'export_reset_pose_bones': True, 'export_rest_position_armature': True,
        'export_pointer_animation': False, 'export_extra_animations': False,
        'export_skins': True, 'export_influence_nb': 4, 'export_all_influences': False, 'export_def_bones': False,
        'export_leaf_bone': False, 'export_armature_object_remove': False, 'export_hierarchy_flatten_bones': False,
        'export_morph': True, 'export_morph_normal': True, 'export_morph_tangent': False, 'export_morph_animation': True,
        'export_lights': False, 'export_cameras': False, 'export_draco_mesh_compression_enable': False,
        'export_gpu_instances': False, 'export_original_specular': False, 'will_save_settings': False,
    }
    required = {'export_format', 'export_tangents', 'export_vertex_color', 'export_yup', 'export_apply',
                'export_animations', 'export_animation_mode', 'export_skins', 'export_influence_nb', 'export_morph',
                'export_lights', 'export_cameras', 'export_draco_mesh_compression_enable', 'export_anim_slide_to_zero'}
    available = set(bpy.ops.export_scene.gltf.get_rna_type().properties.keys())
    missing = sorted(required - available)
    if missing:
        fail('GLTF_EXPORTER', f'exporter lacks required options: {missing}')
    applied = {key: value for key, value in wanted.items() if key in available}
    result = bpy.ops.export_scene.gltf(filepath=filepath, **applied)
    if 'FINISHED' not in result:
        fail('GLTF_EXPORT', 'glTF export failed')
    report['gltfExport'] = {'options': applied, 'unavailableOptions': sorted(set(wanted) - available)}
    return filepath


def decompose_trs(matrix):
    translation = matrix[:3, 3]
    scale = np.linalg.norm(matrix[:3, :3], axis=0)
    rotation = matrix[:3, :3] / scale
    if np.linalg.det(rotation) <= 0 or np.max(np.abs(rotation.T @ rotation - np.identity(3))) > 1e-5:
        fail('BIND_PATCH', 'joint rest transform has shear or a mirror')
    w, x, y, z = rotation_to_quaternion(rotation)
    return translation, np.array([x, y, z, w]), scale


def node_local(node):
    if 'matrix' in node:
        return np.array(node['matrix'], dtype=np.float64).reshape(4, 4).T
    t = node.get('translation', [0.0, 0.0, 0.0])
    x, y, z, w = node.get('rotation', [0.0, 0.0, 0.0, 1.0])
    s = node.get('scale', [1.0, 1.0, 1.0])
    rotation = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    result = np.identity(4)
    result[:3, :3] = rotation * np.array(s)
    result[:3, 3] = t
    return result


def patch_bind_pose(args, filepath, joints, rest):
    # Blender stores bone rest frames as float32 head, tail and roll, which loses up to ~2e-3 rad of
    # roll for bones that point almost exactly along -Y. The exporter copies those frames into the joint
    # nodes' rest TRS and the inverse bind matrices. Replace both with the exact Maya bind pose; the
    # animation channels are unaffected (they come from the sampled pose matrices).
    with open(filepath, 'r', encoding='utf-8') as handle:
        gltf = json.load(handle)
    skin = gltf['skins'][0]
    nodes = gltf['nodes']
    names = [nodes[n]['name'] for n in skin['joints']]
    if sorted(names) != sorted(joints):
        fail('BIND_PATCH', 'skin joints differ from the rebuilt skeleton')
    index = {name: i for i, name in enumerate(joints)}
    world = {name: BLENDER_TO_GLTF @ rest[index[name]] for name in names}
    accessor = gltf['accessors'][skin['inverseBindMatrices']]
    view = gltf['bufferViews'][accessor['bufferView']]
    if (accessor['componentType'] != 5126 or accessor['type'] != 'MAT4' or accessor['count'] != len(names)
            or view.get('byteStride', 64) != 64):
        fail('BIND_PATCH', 'unexpected inverse bind matrix accessor layout')
    offset = view.get('byteOffset', 0) + accessor.get('byteOffset', 0)
    binary = os.path.join(args.output, gltf['buffers'][view['buffer']]['uri'])
    exact = np.array([np.linalg.inv(world[name]) for name in names])
    with open(binary, 'r+b') as handle:
        handle.seek(offset)
        current = np.frombuffer(handle.read(64 * len(names)), dtype='<f4').reshape(-1, 4, 4).transpose(0, 2, 1).astype(np.float64)
        change = float(np.max(np.abs(current - exact)))
        if change > 1e-2:
            fail('BIND_PATCH', f'exported inverse bind matrices differ from the bind pose by {change}')
        handle.seek(offset)
        handle.write(exact.transpose(0, 2, 1).astype('<f4').tobytes())
    parent_of = {child: parent for parent, node in enumerate(nodes) for child in node.get('children', [])}
    joint_nodes = set(skin['joints'])
    translation_change = rotation_change = 0.0
    for node_index in skin['joints']:
        node = nodes[node_index]
        parent = parent_of.get(node_index)
        if parent in joint_nodes:
            parent_world = world[nodes[parent]['name']]
        else:
            parent_world = np.identity(4)
            while parent is not None:
                parent_world = node_local(nodes[parent]) @ parent_world
                parent = parent_of.get(parent)
        local = np.linalg.inv(parent_world) @ world[node['name']]
        previous = node_local(node)
        translation_change = max(translation_change, float(np.linalg.norm(previous[:3, 3] - local[:3, 3])))
        rotation_change = max(rotation_change, rotation_error_radians(previous, local))
        t, q, s = decompose_trs(local)
        for key in ('translation', 'rotation', 'scale', 'matrix'):
            node.pop(key, None)
        if np.max(np.abs(t)) > 0:
            node['translation'] = [float(v) for v in t]
        if abs(q[3]) < 1.0:
            node['rotation'] = [float(v) for v in q]
        if np.max(np.abs(s - 1.0)) > 1e-6:
            node['scale'] = [float(v) for v in s]
    with open(filepath, 'w', encoding='utf-8', newline='\n') as handle:
        handle.write(json.dumps(gltf, indent='\t', separators=(',', ':'), allow_nan=False) + '\n')
    return {'inverseBindMatrixMaxChange': change, 'jointRestTranslationMaxChangeM': translation_change,
            'jointRestRotationMaxChangeRadians': rotation_change}


def check_gltf(args, filepath, report, image_sources):
    with open(filepath, 'r', encoding='utf-8') as handle:
        gltf = json.load(handle)
    problems = []
    if gltf.get('extensionsUsed') or gltf.get('extensionsRequired'):
        problems.append(f'extensions {gltf.get("extensionsUsed")}')
    for mesh in gltf.get('meshes', []):
        for primitive in mesh.get('primitives', []):
            extra = sorted(set(primitive.get('attributes', {})) - ALLOWED_GLTF_ATTRIBUTES)
            if extra:
                problems.append(f'mesh {mesh.get("name")} attributes {extra}')
    for material in gltf.get('materials', []):
        if 'occlusionTexture' in material or material.get('extensions'):
            problems.append(f'material {material.get("name")} has occlusion or extensions')
    uris = [buffer.get('uri', '') for buffer in gltf.get('buffers', [])] + [image.get('uri', '') for image in gltf.get('images', [])]
    for uri in uris:
        if not uri or '/' in uri or '\\' in uri or uri.startswith('.') or ':' in uri:
            problems.append(f'uri {uri!r} is not a plain file name')
    if args.skin and (len(gltf.get('skins', [])) != 1 or len(gltf['skins'][0]['joints']) != report['skeleton']['joints']):
        problems.append('expected one skin with every joint')
    if problems:
        fail('GLTF_CONTRACT', '; '.join(problems))
    expected = sorted({args.name + '.gltf', *uris})
    present = sorted(os.listdir(args.output))
    if present != expected:
        fail('GLTF_FILES', f'output files {present} differ from referenced files {expected}')

    # Exported images: byte-identical copies of the sources, or re-encoded with a known channel mapping.
    textures = gltf.get('textures', [])
    images = gltf.get('images', [])
    roles = {}
    for material in gltf.get('materials', []):
        name = material.get('name')
        pbr = material.get('pbrMetallicRoughness', {})
        for role, info in (('baseColor', pbr.get('baseColorTexture')), ('metallicRoughness', pbr.get('metallicRoughnessTexture')),
                           ('normal', material.get('normalTexture'))):
            if info is not None:
                roles.setdefault(textures[info['index']]['source'], []).append((name, role))
    image_reports = []
    for index, image in enumerate(images):
        path = os.path.join(args.output, image['uri'])
        entry = {'uri': image['uri'], 'bytes': os.path.getsize(path), 'uses': sorted(f'{n}:{r}' for n, r in roles.get(index, []))}
        uses = roles.get(index, [])
        sources = sorted({image_sources[(material, role)] for material, role in uses if (material, role) in image_sources})
        entry['sources'] = sources
        if len(sources) == 1:
            source_path, source_channel = sources[0]
            with open(source_path, 'rb') as a, open(path, 'rb') as b:
                identical = a.read() == b.read()
            entry['byteIdenticalToSource'] = identical
            if not identical:
                entry['pixelCheck'] = compare_pixels(source_path, path, source_channel, {role for _, role in uses})
        image_reports.append(entry)
    for entry in image_reports:
        entry['sources'] = [f'{os.path.basename(path)}:{channel}' for path, channel in entry['sources']]
    report['images'] = image_reports
    return gltf


def load_pixels(path):
    image = bpy.data.images.load(path, check_existing=False)
    try:
        image.colorspace_settings.name = 'Non-Color'
        width, height = image.size
        if width * height > LIMITS['imagePixels']:
            fail('TEXTURE_SIZE', f'{os.path.basename(path)} is too large to compare')
        values = np.empty(width * height * image.channels, dtype=np.float32)
        image.pixels.foreach_get(values)
        return np.rint(values.reshape(height, width, image.channels) * 255.0).astype(np.uint8), image.channels
    finally:
        bpy.data.images.remove(image)


def compare_pixels(source_path, exported_path, source_channel, roles):
    source, _ = load_pixels(source_path)
    exported, exported_channels = load_pixels(exported_path)
    if source.shape[:2] != exported.shape[:2]:
        return {'sizeChanged': [list(source.shape[:2]), list(exported.shape[:2])]}
    result = {'exportedChannels': exported_channels}
    if roles == {'metallicRoughness'}:
        channel = 'rgb'.index(source_channel)
        result['sourceChannelToGreenMismatches'] = int(np.count_nonzero(source[:, :, channel] != exported[:, :, 1]))
        for name, index in (('red', 0), ('blue', 2)):
            plane = exported[:, :, index]
            result[f'{name}Range'] = [int(plane.min()), int(plane.max())]
    else:
        result['rgbMismatches'] = int(np.count_nonzero(np.any(source[:, :, :3] != exported[:, :, :3], axis=2)))
    return result


def gltf_counts(gltf, directory):
    accessors = gltf.get('accessors', [])
    counts = {'meshes': len(gltf.get('meshes', [])), 'primitives': 0, 'vertices': 0, 'triangles': 0,
              'materials': len(gltf.get('materials', [])), 'textures': len(gltf.get('textures', [])),
              'images': len(gltf.get('images', [])), 'nodes': len(gltf.get('nodes', [])),
              'skins': len(gltf.get('skins', [])), 'animations': len(gltf.get('animations', [])), 'morphTargets': 0}
    for mesh in gltf.get('meshes', []):
        for primitive in mesh.get('primitives', []):
            counts['primitives'] += 1
            counts['vertices'] += accessors[primitive['attributes']['POSITION']]['count']
            if 'indices' in primitive:
                counts['triangles'] += accessors[primitive['indices']]['count'] // 3
            counts['morphTargets'] += len(primitive.get('targets', []))
    return counts


# ---------------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------------

def gltf_exporter_version():
    import addon_utils
    for module in addon_utils.modules():
        if module.__name__ == 'io_scene_gltf2':
            return '.'.join(str(part) for part in module.bl_info.get('version', ()))
    return None


def main():
    args = parse_arguments()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    report = {
        'format': REPORT_FORMAT, 'version': 1,
        'tool': {'blender': bpy.app.version_string, 'gltfExporter': gltf_exporter_version()},
        'inputs': {'input': os.path.basename(args.input),
                   'animation': os.path.basename(args.animation) if args.animation else None,
                   'skinMayaAscii': os.path.basename(args.skin_ma) if args.skin_ma else None,
                   'skinClusters': {mesh: cluster for mesh, cluster in args.skin},
                   'armature': args.armature,
                   'materialsMayaAscii': os.path.basename(args.materials_ma) if args.materials_ma else None,
                   'textures': os.path.basename(args.textures) if args.textures else None},
        'settings': {'uvSet': args.uv_set, 'maxInfluences': args.max_influences},
        'fbxImport': {},
    }
    created, applied, unavailable = import_fbx(args.input, not args.skin)
    report['fbxImport']['input'] = {'file': os.path.basename(args.input), 'objects': type_counts(created),
                                    'options': applied, 'unavailableOptions': unavailable}
    log(f'imported {os.path.basename(args.input)}: {type_counts(created)}')
    mesh_reports = {}
    if args.skin:
        targets = {}
        for mesh_name, _ in args.skin:
            obj = bpy.data.objects.get(mesh_name)
            if obj is None or obj.type != 'MESH':
                fail('SKIN_MESH', f'{mesh_name!r} is not a mesh object of --input')
            targets[mesh_name] = obj
        for obj in targets.values():
            world = obj.matrix_world.copy()
            obj.parent = None
            obj.matrix_world = world
        remove_objects([obj for obj in bpy.data.objects if obj.name not in targets])
        for action in list(bpy.data.actions):
            bpy.data.actions.remove(action)
        purge_orphans()
        mesh_reports, skin_result = rebuild_skin(args, scene, targets, report)
        meshes = list(targets.values())
    else:
        skin_result = None
        meshes = [obj for obj in bpy.data.objects if obj.type == 'MESH']
        for obj in meshes:
            if len(obj.data.vertices) > LIMITS['meshVertices']:
                fail('MESH_SIZE', f'{obj.name} has too many vertices')
            ngons, normal_error = bake_transform_and_triangulate(obj, Matrix.Identity(4))
            mesh_reports[obj.name] = {'polygonsTriangulated': ngons, 'customNormalMaxAngleChangeRadians': normal_error}
    for obj in meshes:
        if len(obj.data.vertices) > LIMITS['meshVertices']:
            fail('MESH_SIZE', f'{obj.name} has too many vertices')

    # Attribute cleanup: one UV set, no colour attributes (COLOR_0 would tint the base colour).
    for obj in meshes:
        mesh = obj.data
        entry = mesh_reports.setdefault(obj.name, {})
        uv_names = [layer.name for layer in mesh.uv_layers]
        if args.uv_set not in uv_names:
            fail('UV_SET', f'{obj.name} has no UV set {args.uv_set!r} (has {uv_names})')
        for name in uv_names:
            if name != args.uv_set:
                mesh.uv_layers.remove(mesh.uv_layers[name])
        mesh.uv_layers.active = mesh.uv_layers[args.uv_set]
        mesh.uv_layers[args.uv_set].active_render = True
        colors = []
        for attribute in list(mesh.color_attributes):
            sample = set()
            for item in attribute.data:
                sample.add(tuple(round(component, 3) for component in item.color))
                if len(sample) > 16:
                    break
            colors.append({'name': attribute.name, 'domain': attribute.domain, 'type': attribute.data_type,
                           'distinctValuesUpTo16': len(sample),
                           'values': sorted(list(value) for value in sample) if len(sample) <= 4 else None})
            mesh.color_attributes.remove(attribute)
        for key in list(obj.keys()):
            del obj[key]
        for key in list(mesh.keys()):
            del mesh[key]
        entry.update({'uvSetsInSource': uv_names, 'uvSetKept': args.uv_set, 'colorAttributesDropped': colors,
                      'customNormals': bool(mesh.has_custom_normals),
                      'shapeKeys': len(mesh.shape_keys.key_blocks) - 1 if mesh.shape_keys else 0,
                      'vertices': len(mesh.vertices), 'triangles': sum(len(p.vertices) - 2 for p in mesh.polygons)})

    materials = []
    for obj in meshes:
        for slot in obj.material_slots:
            if slot.material is not None and slot.material not in materials:
                materials.append(slot.material)
    image_sources = {}
    if args.materials_ma:
        materials_started = time.perf_counter()
        _, specs = rebuild_materials(args, materials, report)
        for material in materials:
            spec = specs.get(material.name)
            if spec is None:
                continue
            for plug, texture in spec['inputs'].items():
                role = {'dc': 'baseColor', 'rgh': 'metallicRoughness', 'mtl': 'metallicRoughness', 'nrm': 'normal'}[plug]
                image_sources[(short_name(material.name), role)] = (texture['path'], 'rgb' if texture['channel'] == 'rgb' else texture['channel'])
        log(f'materials rebuilt ({time.perf_counter() - materials_started:.2f}s)')

    # Names without namespaces.
    for obj in meshes:
        obj.data.name = short_name(obj.name)
    renamed = {}
    for block in [*meshes, *materials]:
        name = short_name(block.name)
        renamed.setdefault(type(block).__name__, set())
        if name in renamed[type(block).__name__]:
            fail('NAMES', f'{block.name} collides with another name after removing namespaces')
        renamed[type(block).__name__].add(name)
        block.name = name
    report['meshes'] = [dict({'name': short_name(name)}, **entry) for name, entry in sorted(mesh_reports.items(), key=lambda item: item[0])]
    purge_orphans()

    os.mkdir(args.output)
    export_started = time.perf_counter()
    filepath = export_gltf(args, report)
    log(f'exported {os.path.basename(filepath)} ({time.perf_counter() - export_started:.2f}s)')
    if skin_result is not None:
        report['bindPosePatch'] = patch_bind_pose(args, filepath, skin_result['joints'], skin_result['rest'])
        log(f'bind pose patched: {report["bindPosePatch"]}')
    check_started = time.perf_counter()
    gltf = check_gltf(args, filepath, report, image_sources)
    report['counts'] = gltf_counts(gltf, args.output)
    log(f'output checked ({time.perf_counter() - check_started:.2f}s): {report["counts"]}')
    with open(os.path.join(args.output, REPORT_NAME), 'x', encoding='utf-8') as handle:
        handle.write(json.dumps(rounded(report), indent=2) + '\n')
    log('done')


try:
    main()
except ConversionError as error:
    print(f'CONVERT_FBX_GLTF_ERROR {error.code}: {error}', file=sys.stderr, flush=True)
    sys.exit(2)
except SystemExit:
    raise
except BaseException:
    traceback.print_exc()
    print('CONVERT_FBX_GLTF_ERROR UNEXPECTED', file=sys.stderr, flush=True)
    sys.exit(3)
