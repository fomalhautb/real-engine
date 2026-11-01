#!/usr/bin/env node
// Publishes the output of scripts/convert-fbx-gltf.py as a derived asset directory.
//   node scripts/write-gltf-conversion-manifest.mjs --conversion blender-output-directory --output new-directory
//     --source-dir extracted-source-directory --archive source.zip --license credits_license.txt
//     --source ROLE=relative/path [--source ...] [--not-carried-over notes.json]
// The conversion directory must hold exactly <name>.gltf, its buffers and images (plain file names) and
// conversion-report.json. Every --source path is relative to --source-dir and is hashed; ROLE is one of
// input, animation, skinMayaAscii, materialsMayaAscii, textures (a directory: the textures the report
// names are hashed). The staged copy gets the license, then manifest.json last; the directory is
// published with publishDirectory (manifest linked last). The manifest records source, script, tool and
// output hashes, export options and counts read from the glTF and its accessors. No timestamps or
// absolute paths are written.
import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { copyFile, lstat, mkdir, mkdtemp, readdir, readFile, realpath, writeFile } from 'node:fs/promises';
import { basename, dirname, isAbsolute, join, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { exists, publishDirectory, removeOwnTemporary } from './asset-output.mjs';

const SCRIPT = fileURLToPath(import.meta.url);
const CONVERTER = join(dirname(SCRIPT), 'convert-fbx-gltf.py');
const MiB = 1024 * 1024;
const LIMITS = Object.freeze({ jsonBytes: 64 * MiB, binBytes: 1024 * MiB, imageBytes: 256 * MiB, files: 256,
  licenseBytes: 4 * MiB, notesBytes: 1 * MiB, archiveBytes: 8192 * MiB, sourceBytes: 2048 * MiB });
const ROLES = new Set(['input', 'animation', 'skinMayaAscii', 'materialsMayaAscii', 'textures']);
const ROLE_FLAGS = { input: '--input', animation: '--animation', skinMayaAscii: '--skin-ma',
  materialsMayaAscii: '--materials-ma', textures: '--textures' };
const COMPONENTS = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT4: 16 };
const COMPONENT_BYTES = { 5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4 };
const REPORT = 'conversion-report.json';
const USAGE = 'usage: write-gltf-conversion-manifest --conversion dir --output new-dir --source-dir dir --archive zip '
  + '--license file --source ROLE=path [--source ...] [--not-carried-over notes.json]';

const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const jsonBytes = value => Buffer.from(JSON.stringify(value, null, 2) + '\n');
const byFile = (a, b) => (a.file < b.file ? -1 : a.file > b.file ? 1 : 0);
function fail(code, detail) { throw new Error(detail === undefined ? code : `${code}: ${detail}`); }

function argumentsOf(argv) {
  const single = new Set(['--conversion', '--output', '--source-dir', '--archive', '--license', '--not-carried-over']);
  const values = new Map(), sources = [];
  for (let i = 0; i < argv.length; i += 2) {
    const key = argv[i], value = argv[i + 1];
    if (value === undefined) fail('MANIFEST_ARGUMENTS', USAGE);
    if (key === '--source') {
      const separator = value.indexOf('=');
      const role = value.slice(0, separator), path = value.slice(separator + 1);
      if (separator < 1 || !ROLES.has(role) || !path || isAbsolute(path) || path.split(/[\\/]/).includes('..')) fail('MANIFEST_SOURCE', value);
      if (sources.some(source => source.role === role)) fail('MANIFEST_SOURCE_DUPLICATE', role);
      sources.push({ role, path: path.split(sep).join('/') });
    } else if (single.has(key) && !values.has(key)) values.set(key, value);
    else fail('MANIFEST_ARGUMENTS', USAGE);
  }
  for (const key of ['--conversion', '--output', '--source-dir', '--archive', '--license']) if (!values.has(key)) fail('MANIFEST_ARGUMENTS', USAGE);
  if (!sources.some(source => source.role === 'input')) fail('MANIFEST_SOURCE', 'an input source is required');
  return { conversion: resolve(values.get('--conversion')), output: resolve(values.get('--output')),
    sourceDir: resolve(values.get('--source-dir')), archive: resolve(values.get('--archive')),
    license: resolve(values.get('--license')), notes: values.has('--not-carried-over') ? resolve(values.get('--not-carried-over')) : null, sources };
}

async function readRegular(path, limit, code) {
  const info = await lstat(path).catch(error => fail(code + '_MISSING', `${basename(path)} (${error.code})`));
  if (!info.isFile() || info.size > limit) fail(code, `${basename(path)} is not a regular file within ${limit} bytes`);
  const bytes = await readFile(path);
  if (bytes.length !== info.size) fail(code + '_CHANGED', basename(path));
  return bytes;
}

async function hashStream(path, limit, code) {
  const info = await lstat(path).catch(error => fail(code + '_MISSING', `${basename(path)} (${error.code})`));
  if (!info.isFile() || info.size > limit) fail(code, `${basename(path)} is not a regular file within ${limit} bytes`);
  const digest = createHash('sha256');
  let bytes = 0;
  for await (const chunk of createReadStream(path, { highWaterMark: 4 * MiB })) { digest.update(chunk); bytes += chunk.length; }
  if (bytes !== info.size) fail(code + '_CHANGED', basename(path));
  return { bytes, sha256: digest.digest('hex') };
}

function plainName(uri, code) {
  if (typeof uri !== 'string' || !/^[A-Za-z0-9][A-Za-z0-9_.-]*$/.test(uri) || uri === REPORT || uri === 'manifest.json') fail(code, String(uri));
  return uri;
}

function accessorView(doc, buffers, index) {
  const accessor = doc.accessors?.[index];
  if (!accessor || accessor.sparse) fail('MANIFEST_ACCESSOR', `accessor ${index} is missing or sparse`);
  const view = doc.bufferViews?.[accessor.bufferView];
  const size = COMPONENT_BYTES[accessor.componentType], components = COMPONENTS[accessor.type];
  if (!view || !size || !components) fail('MANIFEST_ACCESSOR', `accessor ${index} layout`);
  const stride = view.byteStride ?? size * components;
  const start = (view.byteOffset ?? 0) + (accessor.byteOffset ?? 0);
  const bytes = buffers[view.buffer];
  if (!bytes || start + stride * (accessor.count - 1) + size * components > (view.byteOffset ?? 0) + view.byteLength
    || (view.byteOffset ?? 0) + view.byteLength > bytes.length) fail('MANIFEST_ACCESSOR', `accessor ${index} exceeds its buffer view`);
  const data = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const read = { 5120: o => data.getInt8(o), 5121: o => data.getUint8(o), 5122: o => data.getInt16(o, true),
    5123: o => data.getUint16(o, true), 5125: o => data.getUint32(o, true), 5126: o => data.getFloat32(o, true) }[accessor.componentType];
  return { accessor, count: accessor.count, components, value: (element, component) => read(start + element * stride + component * size) };
}

function countsOf(doc, buffers) {
  const counts = { meshes: doc.meshes?.length ?? 0, primitives: 0, vertices: 0, triangles: 0, materials: doc.materials?.length ?? 0,
    textures: doc.textures?.length ?? 0, images: doc.images?.length ?? 0, nodes: doc.nodes?.length ?? 0,
    skins: doc.skins?.length ?? 0, jointsPerSkin: (doc.skins ?? []).map(skin => skin.joints.length),
    maxInfluencesPerVertex: 0, influenceHistogram: {}, animations: doc.animations?.length ?? 0, morphTargets: 0 };
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  for (const mesh of doc.meshes ?? []) {
    for (const primitive of mesh.primitives) {
      if ((primitive.mode ?? 4) !== 4) fail('MANIFEST_PRIMITIVE', 'only triangle lists are expected');
      counts.primitives += 1;
      const positions = accessorView(doc, buffers, primitive.attributes.POSITION);
      counts.vertices += positions.count;
      for (let axis = 0; axis < 3; axis += 1) {
        min[axis] = Math.min(min[axis], positions.accessor.min[axis]);
        max[axis] = Math.max(max[axis], positions.accessor.max[axis]);
      }
      counts.triangles += primitive.indices === undefined ? positions.count / 3 : accessorView(doc, buffers, primitive.indices).count / 3;
      counts.morphTargets += primitive.targets?.length ?? 0;
      if (primitive.attributes.WEIGHTS_0 !== undefined) {
        if (Object.keys(primitive.attributes).some(name => /^(JOINTS|WEIGHTS)_[1-9]/.test(name))) fail('MANIFEST_SKIN', 'more than one influence set');
        const weights = accessorView(doc, buffers, primitive.attributes.WEIGHTS_0);
        const joints = accessorView(doc, buffers, primitive.attributes.JOINTS_0);
        const skinJoints = Math.max(...counts.jointsPerSkin);
        for (let vertex = 0; vertex < weights.count; vertex += 1) {
          let used = 0;
          for (let k = 0; k < 4; k += 1) {
            if (weights.value(vertex, k) > 0) {
              used += 1;
              if (joints.value(vertex, k) >= skinJoints) fail('MANIFEST_SKIN', `joint index outside the skin in ${mesh.name}`);
            }
          }
          counts.influenceHistogram[used] = (counts.influenceHistogram[used] ?? 0) + 1;
          counts.maxInfluencesPerVertex = Math.max(counts.maxInfluencesPerVertex, used);
        }
      }
    }
  }
  counts.boundsMeters = { min, max };
  counts.animationDetails = (doc.animations ?? []).map(animation => {
    const paths = {}, interpolation = {};
    let start = Infinity, end = -Infinity, keyframesMax = 0;
    for (const channel of animation.channels) paths[channel.target.path] = (paths[channel.target.path] ?? 0) + 1;
    for (const sampler of animation.samplers) {
      const input = accessorView(doc, buffers, sampler.input);
      interpolation[sampler.interpolation ?? 'LINEAR'] = (interpolation[sampler.interpolation ?? 'LINEAR'] ?? 0) + 1;
      start = Math.min(start, input.value(0, 0));
      end = Math.max(end, input.value(input.count - 1, 0));
      keyframesMax = Math.max(keyframesMax, input.count);
    }
    return { name: animation.name ?? null, channels: animation.channels.length, channelsByPath: paths,
      samplers: animation.samplers.length, interpolation, startSeconds: start, endSeconds: end,
      durationSeconds: end - start, keyframesMax,
      targetNodes: new Set(animation.channels.map(channel => channel.target.node)).size };
  });
  return counts;
}

async function main() {
  const config = argumentsOf(process.argv.slice(2));
  if (await exists(config.output)) fail('MANIFEST_OUTPUT_EXISTS', config.output);
  const reportBytes = await readRegular(join(config.conversion, REPORT), LIMITS.jsonBytes, 'MANIFEST_REPORT');
  const report = JSON.parse(reportBytes);
  if (report.format !== 'real-engine-fbx-gltf-conversion' || report.version !== 1) fail('MANIFEST_REPORT', 'unexpected report format');

  // The glTF and exactly the files it references.
  const entries = (await readdir(config.conversion, { withFileTypes: true })).map(entry => {
    if (!entry.isFile()) fail('MANIFEST_CONVERSION', `${entry.name} is not a regular file`);
    return entry.name;
  }).sort();
  if (entries.length > LIMITS.files) fail('MANIFEST_CONVERSION', 'too many files');
  const gltfNames = entries.filter(name => name.endsWith('.gltf'));
  if (gltfNames.length !== 1) fail('MANIFEST_CONVERSION', 'expected exactly one .gltf');
  const gltfBytes = await readRegular(join(config.conversion, gltfNames[0]), LIMITS.jsonBytes, 'MANIFEST_GLTF');
  const doc = JSON.parse(gltfBytes);
  const buffers = [];
  for (const buffer of doc.buffers ?? []) {
    const bytes = await readRegular(join(config.conversion, plainName(buffer.uri, 'MANIFEST_BUFFER_URI')), LIMITS.binBytes, 'MANIFEST_BUFFER');
    if (bytes.length < buffer.byteLength) fail('MANIFEST_BUFFER', `${buffer.uri} is shorter than byteLength`);
    buffers.push(bytes);
  }
  const imageUris = (doc.images ?? []).map(image => plainName(image.uri, 'MANIFEST_IMAGE_URI'));
  const referenced = [gltfNames[0], ...(doc.buffers ?? []).map(buffer => buffer.uri), ...imageUris, REPORT].sort();
  if (JSON.stringify([...new Set(referenced)].sort()) !== JSON.stringify(entries)) {
    fail('MANIFEST_CONVERSION', `files ${entries.join(',')} differ from referenced ${referenced.join(',')}`);
  }
  const counts = countsOf(doc, buffers);

  // Sources, hashed in place.
  const sourceDir = await realpath(config.sourceDir);
  const extraction = await readRegular(join(sourceDir, 'manifest.json'), LIMITS.jsonBytes, 'MANIFEST_EXTRACTION');
  const extractionManifest = JSON.parse(extraction);
  const archive = await hashStream(config.archive, LIMITS.archiveBytes, 'MANIFEST_ARCHIVE');
  const recordedArchive = extractionManifest.archive?.sha256 ?? extractionManifest.source?.sha256;
  if (recordedArchive && recordedArchive !== archive.sha256) fail('MANIFEST_ARCHIVE', 'archive hash differs from the extraction manifest');
  const sourceFiles = [];
  const reportInputs = { input: report.inputs.input, animation: report.inputs.animation, skinMayaAscii: report.inputs.skinMayaAscii,
    materialsMayaAscii: report.inputs.materialsMayaAscii, textures: report.inputs.textures };
  for (const source of config.sources) {
    const path = resolve(sourceDir, source.path);
    if (relative(sourceDir, path).startsWith('..')) fail('MANIFEST_SOURCE', source.path);
    if (basename(path) !== reportInputs[source.role]) fail('MANIFEST_SOURCE', `${source.role} ${source.path} is not the report's ${reportInputs[source.role]}`);
    if (source.role === 'textures') {
      const used = new Set((report.images ?? []).flatMap(image => image.sources.map(entry => entry.split(':')[0])));
      for (const file of [...used].sort()) {
        sourceFiles.push({ role: 'texture', path: `${source.path}/${file}`, ...(await hashStream(join(path, file), LIMITS.sourceBytes, 'MANIFEST_SOURCE')) });
      }
    } else {
      sourceFiles.push({ role: source.role, path: source.path, ...(await hashStream(path, LIMITS.sourceBytes, 'MANIFEST_SOURCE')) });
    }
  }
  for (const [role, file] of Object.entries(reportInputs)) {
    if (file && !config.sources.some(source => source.role === role)) fail('MANIFEST_SOURCE', `the report names ${role} ${file}; pass --source ${role}=...`);
  }
  const licenseBytes = await readRegular(config.license, LIMITS.licenseBytes, 'MANIFEST_LICENSE');
  const licenseName = basename(config.license);
  if (referenced.includes(licenseName)) fail('MANIFEST_LICENSE', 'license name collides with an output file');
  const notes = config.notes ? JSON.parse(await readRegular(config.notes, LIMITS.notesBytes, 'MANIFEST_NOTES')) : null;
  if (notes !== null && !Array.isArray(notes)) fail('MANIFEST_NOTES', 'expected a JSON array');

  const command = ['blender', '--background', '--factory-startup', '--disable-autoexec', '--threads', '1',
    '--python', 'scripts/convert-fbx-gltf.py', '--',
    ...config.sources.flatMap(source => [ROLE_FLAGS[source.role], source.path]),
    '--output', '<new directory>', '--name', gltfNames[0].replace(/\.gltf$/, ''),
    ...(report.inputs.armature ? ['--armature', report.inputs.armature] : []),
    ...Object.entries(report.inputs.skinClusters ?? {}).flatMap(([mesh, cluster]) => ['--skin', `${mesh}=${cluster}`]),
    ...(report.settings.uvSet !== 'map1' ? ['--uv-set', report.settings.uvSet] : []),
    ...(report.settings.maxInfluences !== 4 ? ['--max-influences', String(report.settings.maxInfluences)] : [])];

  const parent = dirname(config.output);
  await mkdir(parent, { recursive: true });
  const temporary = await mkdtemp(join(parent, `.${basename(config.output)}.tmp-`));
  try {
    const files = [];
    for (const name of entries) {
      const bytes = await readRegular(join(config.conversion, name), name.endsWith('.png') ? LIMITS.imageBytes : LIMITS.binBytes, 'MANIFEST_COPY');
      await writeFile(join(temporary, name), bytes, { flag: 'wx' });
      files.push({ file: name, bytes: bytes.length, sha256: sha256(bytes) });
    }
    await writeFile(join(temporary, licenseName), licenseBytes, { flag: 'wx' });
    const license = { file: licenseName, bytes: licenseBytes.length, sha256: sha256(licenseBytes) };
    files.push(license);
    const manifest = {
      version: 1, format: 'real-engine-gltf-conversion',
      purpose: 'glTF 2.0 conversion of an FBX character with skin and animation for engine import; see conversion-report.json for every check',
      tool: {
        converter: { script: 'scripts/convert-fbx-gltf.py', sha256: sha256(await readFile(CONVERTER)) },
        finaliser: { script: 'scripts/write-gltf-conversion-manifest.mjs', sha256: sha256(await readFile(SCRIPT)) },
        blender: report.tool.blender, gltfExporter: report.tool.gltfExporter, command: command.join(' '),
        determinism: 'byte-identical across runs with --threads 1; multithreaded MikkTSpace can flip a 1e-4 TANGENT rounding step',
      },
      source: {
        archive: { file: basename(config.archive), ...archive },
        extraction: { directory: basename(sourceDir), manifestSha256: sha256(extraction) },
        files: sourceFiles,
      },
      gltf: { file: gltfNames[0], buffers: (doc.buffers ?? []).map(buffer => buffer.uri), images: imageUris,
        generator: doc.asset?.generator ?? null, extensionsUsed: doc.extensionsUsed ?? [] },
      importOptions: report.fbxImport,
      exportOptions: report.gltfExport,
      counts,
      conversion: {
        report: files.find(file => file.file === REPORT),
        skeleton: report.skeleton ? { joints: report.skeleton.joints, root: report.skeleton.root, restPose: report.skeleton.restPose } : null,
        bindPosePatch: report.bindPosePatch ?? null,
        animation: report.animation ? { name: report.animation.name, sourceAction: report.animation.sourceAction,
          frames: report.animation.frames, fps: report.animation.fps, timeMapping: report.animation.timeMapping } : null,
      },
      notCarriedOver: notes,
      license,
      files: files.sort(byFile),
    };
    await writeFile(join(temporary, 'manifest.json'), jsonBytes(manifest), { flag: 'wx' });
    await publishDirectory(temporary, config.output, { hardLinkFiles: true });
    process.stdout.write(JSON.stringify({ output: basename(config.output), files: files.length, counts: { ...counts, animationDetails: undefined } }) + '\n');
  } finally { await removeOwnTemporary(temporary, parent, config.output); }
}

try { await main(); }
catch (error) { process.stderr.write(`write-gltf-conversion-manifest: ${error.message}\n`); process.exitCode = 1; }
