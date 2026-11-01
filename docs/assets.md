# Third-party test assets

## Small glTF fixtures

These are unmodified glTF 2.0 fixtures from [Khronos glTF Sample Assets](https://github.com/KhronosGroup/glTF-Sample-Assets), pinned at commit [`edc7c9e67c639d230715049ee31f9a96a6babbbe`](https://github.com/KhronosGroup/glTF-Sample-Assets/commit/edc7c9e67c639d230715049ee31f9a96a6babbbe). Each `glTF/` directory preserves the upstream `.gltf` and every referenced external `.bin` file. The upstream `README.md`, `metadata.json`, and `LICENSE.md` are copied alongside it. Files are byte-for-byte identical to that commit; their Git blob hashes were checked against the upstream contents API.

| Fixture | Entry point | Loader coverage | Upstream source |
| --- | --- | --- | --- |
| Simple Skin | `/assets/gltf/SimpleSkin/glTF/SimpleSkin.gltf` | 3 nodes, one mesh primitive with 10 vertices and 8 triangles, one skin with 2 joints and inverse bind matrices, `JOINTS_0`/`WEIGHTS_0`, one LINEAR joint rotation animation with 12 keys from 0 to 5.5 s. No declared material, normal, or texture; glTF defaults apply. | [Model directory](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/edc7c9e67c639d230715049ee31f9a96a6babbbe/Models/SimpleSkin) |
| Animated Morph Cube | `/assets/gltf/AnimatedMorphCube/glTF/AnimatedMorphCube.gltf` | 1 node, one mesh primitive with 24 vertices and 12 triangles, 2 morph targets (`thin`, `angle`) with position/normal/tangent deltas, one LINEAR weights animation with 127 keys from 0 to approximately 4.2 s. One metallic-roughness material with factors only; no textures. | [Model directory](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/edc7c9e67c639d230715049ee31f9a96a6babbbe/Models/AnimatedMorphCube) |

The model data and binaries are [CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/legalcode). The accompanying metadata files are [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/legalcode), as specified in each copied `LICENSE.md`. Upstream credits: Simple Skin © 2017 Public, Marco Hutter for everything; Animated Morph Cube © 2017 Public, Microsoft for everything. Trademarks and logos are excluded by the upstream license files. No screenshots or logos are included here.

## SHA-256 manifest

Hashes cover the exact copied bytes. Paths are relative to `public/assets/gltf/`.

| Path | SHA-256 |
| --- | --- |
| `SimpleSkin/glTF/SimpleSkin.gltf` | `8d92e9888340eb98e82a65a1f8b037d9a1d9f09e5fd0b6ba6d40fd93700b3239` |
| `SimpleSkin/glTF/SimpleSkin_animation.bin` | `ff21b1bc3d0abbe53f7665060ec6c22d873b6e85a7669171a2b6b8ae01c39347` |
| `SimpleSkin/glTF/SimpleSkin_geometry.bin` | `35f0ca6bc07976c0aadc163d4737758989cfcc9870a99800004c4478fe698b63` |
| `SimpleSkin/glTF/SimpleSkin_inverseBindMatrices.bin` | `5844ab221cd2ad367420a248f6387b46e5f8498604427730a37ee96b4d4ec599` |
| `SimpleSkin/glTF/SimpleSkin_skinningData.bin` | `413bc8a8c0da673df767874e6f05f0b00c9261756a08586b9d555bd5535414b5` |
| `SimpleSkin/LICENSE.md` | `92caa6f47943c19aaaf413af6e2c201d8dbeebf91e643166edf63b262f988c38` |
| `SimpleSkin/README.md` | `6545a0f376b1418c1b2fb3a54706c828c60239a6d9483e03276dca24ebdcc996` |
| `SimpleSkin/metadata.json` | `4b55cd46f47884119db596f29fc8026ac01af65009503495abf380d0f9bc8600` |
| `AnimatedMorphCube/glTF/AnimatedMorphCube.gltf` | `0b910ed4b52fd9fbb565911fc7f9f285edb0f30fa4913a93143466d439d1092e` |
| `AnimatedMorphCube/glTF/AnimatedMorphCube.bin` | `4aba93918adba56c1cc688a38c527bc647e26f2c386802e119b4d01b7250dd23` |
| `AnimatedMorphCube/LICENSE.md` | `158a0614b84cd26077483730bd105ee62d5a8db2a7e4267db911c77a2ee0717c` |
| `AnimatedMorphCube/README.md` | `b872b016a274f02a0060117976150b1180f4090aa5042e1b5861d5d178d0971d` |
| `AnimatedMorphCube/metadata.json` | `53ec35a6148f1fa25936f092a050d821590f597cc3f67050e1b3a9812be4e572` |

## Local scene assets (2026-10-22)

The baseline scene is the existing **Intel new Sponza main building without add-ons**. Later scale tests add the Intel packages below and Bistro. Classic Sponza is not an additional integration dependency. Large source archives, converted payloads, and download records stay under ignored `artifacts/`; they are not distributed with this repository or included in ordinary CI.

The five Intel add-ons have completed downloads and independent local SHA-256 rechecks. Their stable download links come from the [Intel Samples Library](https://www.intel.com/content/www/us/en/developer/topic-technology/graphics-research/samples.html). Per-package JSON records in `artifacts/large-scenes-downloads/intel-sponza/` preserve source URLs, byte lengths, hashes and response metadata; `results.json` records completion. ZIP directory and small metadata inspection did not extract the model payloads or validate every compressed entry.

| Package | Archive bytes | SHA-256 | Source |
| --- | ---: | --- | --- |
| Curtains | 786,898,766 | `3ba96e967c8f5ad0a133309cedb342e3563f9cccb42d04e188f55c0f2125bb65` | [Intel 726650](https://cdrdv2.intel.com/v1/dl/getContent/726650) |
| Ivy | 1,383,921,436 | `e1e2dfacf3db023e8674092fbe81f0d47e361d4ecf1d798802b23ed5d8d4c052` | [Intel 726656](https://cdrdv2.intel.com/v1/dl/getContent/726656) |
| Trees | 951,147,802 | `72263ee9e26b96b72e86718c900274f6acfd767097806c03fbb36177ecd8e6d7` | [Intel 726662](https://cdrdv2.intel.com/v1/dl/getContent/726662) |
| Emissive Candles | 3,190,731,713 | `f8a43d972f377e7eb25e52fdc92faed425ad001a4516e0ecef436ff2f8663396` | [Intel 726676](https://cdrdv2.intel.com/v1/dl/getContent/726676) |
| Animated Knight | 1,202,508,298 | `9112d9789ab2da50c77529907833bd008e5fa602f89438a5c1e82d7d4bcde2a5` | [Intel 763175](https://cdrdv2.intel.com/v1/dl/getContent/763175) |
| ORCA Bistro v5.2 | 894,377,473 | `0d50e3c724c6c5da19f8eb99ad3f53e36fec37ffa2df9621f9ccf0603f3934e1` | [NVIDIA download](https://developer.nvidia.com/bistro) |

The first four Intel add-ons contain glTF, FBX and USD variants. Animated Knight contains Alembic, FBX, USD and Maya data, with no glTF entry; on 2026-10-31 it was converted to glTF with its skin and animation (see [Local conversions](#local-conversions-2026-10-3031-apple-m3-max)). All five preserve the same `credits_license.txt`, SHA-256 `3f981e31f8de04f3754e63d7c259a0a78f71db8f01b5155c58cac52fb79cc2b9`. That file includes specific use and credit notices before the CC BY 4.0 text. Preserve the complete notice with derived assets; the CC0 statement above applies only to the two small fixtures.

Bistro is the original ORCA source referenced by the [Bevy large-scenes example](https://github.com/bevyengine/bevy/tree/main/examples/large_scenes/bistro), not the separately processed glTF linked from Discord. Its completed archive, SHA-256 record and ZIP inspection are in `artifacts/large-scenes-downloads/bistro-orca/`. The directory contains 642 files, including three FBX scenes, 622 DDS textures, an HDR environment and two Falcor `.pyscene` files; declared unpacked size is 1,719,121,581 bytes. On 2026-10-30 all 642 files were extracted with `scripts/extract-archive.mjs`, which checks every entry's CRC32 and records each file's SHA-256 in the extraction manifest; conversion status is under [Local conversions](#local-conversions-2026-10-3031-apple-m3-max). The archive preserves `LICENSE.txt` (CC BY 4.0) and `README.txt` with Amazon Lumberyard / ORCA credit instructions; see the [original release](https://developer.nvidia.com/orca/amazon-lumberyard-bistro).

Its README distinguishes unique from instance-expanded triangle counts: interior 1,044,183 / 1,046,609; interior with wine 935,853 / 1,293,691; exterior 2,799,704 / 2,832,120. These are publisher counts, not this engine's measured draw or residency counts. Conversion must preserve opacity, the packed occlusion/roughness/metalness texture channels, DirectX normal-map convention, emissive materials, and the additional wine-scene material parameters. Download completion is not material, animation, lighting or scene acceptance.

## Local conversions (2026-10-30/31, Apple M3 Max)

Converted payloads and evidence live in ignored `artifacts/` directories with a `-mac-20261030` or `-mac-20261031` suffix. A directory is complete once its `manifest.json` exists and is never rewritten. Every derived directory keeps the source license files. None of these bytes are committed.

Tools:

- `scripts/extract-archive.mjs`: verified ZIP extraction.
- `scripts/split-gltf-primitives.mjs`: deterministic split of oversized primitives.
- `scripts/build-gltf-geometry.mjs`: geometry pages.
- `scripts/audit-geometry-boundaries.mjs`: exact crack audit of built pages.
- `scripts/build-gltf-textures.mjs --encoder builtin`: deterministic BC7, with normal maps on the RGBA8 linear path.
- `scripts/audit-gltf-uv-demand.mjs`.
- `scripts/write-asset-set.mjs`: binds the packages of one source, optionally with a camera path. With `--compose` it binds finished sets of several sources.
- FBX scenes: `scripts/blender/convert-fbx-scene.py`, followed by `scripts/apply-fbx-material-map.mjs`.
- The skinned character: `scripts/convert-fbx-gltf.py`, `scripts/write-gltf-conversion-manifest.mjs` and `scripts/validate-gltf-skinned-animation.mjs`.

**Intel Sponza main.** `Main_sponza.zip` (3,987,608,266 bytes, SHA-256 `b8bb853884ab1566b3beb35666bd09882a4e0dc16661e4684e103792cf0229b9`): 84 files extracted. Every derived package is bound to `NewSponza_Main_glTF_003.gltf`, SHA-256 `e04c4c540c74bdddcbd3f590a85c14119bfd5839702246a23b3a737c0cce2400`.

**Intel add-ons.** The counts below come from the glTF audits and count accessor triangles.

| Package | Primitives | Triangles | Status |
| --- | ---: | ---: | --- |
| Curtains | 43 | 1,997,366 | Built: geometry pages (group 16), 8 textures, UV audit and asset set. |
| Trees | 4 | 2,023,747 | The group-16 build stopped at `GEOMETRY_QUERY_PROXY_TRIANGLE_BUDGET`. Built: group-8 pages, 4 textures, UV audit and asset set. `TEXCOORD_1` is present but unused. The leaves are omitted from paging: 2 primitives with material `LeafSpring` (BLEND), 1,354,570 triangles. |
| Ivy | 2 → 17 | 5,065,558 | Split offline into 17 primitives. Built: geometry pages (groups 8 and 16), 3 textures, UV audit and asset set. |
| Emissive Candles | 3 | 19,540,000 | Not converted: 10,002 nodes, 648,098,592 declared buffer bytes, `KHR_lights_punctual` required. |

**Ivy split.** Mesh 0 primitive 0 has 4,905,636 triangles, above the geometry builder's limit of 1,000,000 triangles or vertices per primitive.

- *Split.* `scripts/split-gltf-primitives.mjs --max-triangles 320000 --max-vertices 1000000` cuts it into 16 parts by a recursive median split of triangle centroids. Each part has 306,602 or 306,603 triangles and 192k–204k vertices. Mesh 1 primitive 0 (159,922 triangles) is copied unsplit.
- *What is preserved.* Parts keep source triangle and corner order, materials, UVs, normals and tangents byte for byte. The 40,369 vertices on a cut (1.30%) are duplicated exactly, so each new border is a shared seam that the builder locks. 5,065,558 triangles go in and come out.
- *Provenance.* The output manifest records the split tool's hash, so a changed tool needs a new output directory.

| Directory | Manifest SHA-256 | Contents |
| --- | --- | --- |
| `sponza-ivy-split-mac-20261031` | `43b8cf3d3ec629bc8023d9fd46cf95c962c0a9ac5e99d749e4c9e78f98cc74f2` | Split glTF (bin 272,562,568 bytes), 3 hard-linked images, per-part provenance maps. |
| `sponza-ivy-group16-pages-mac-20261031` | `cb6aca9b43964dfb560e9f595956dc2e19cde9576418ab3dd5c546860acfa3e5` | 152,895 clusters, 19,781 pages, 738.8 MB encoded / 1,210.6 MB decoded, roots 81.4 MB decoded. Proxies 543,482 triangles; the largest has 74,204 of the 262,144 limit. Used by the Ivy set. |
| `sponza-ivy-group8-pages-mac-20261031` | `1fcb4fb68fdd8f53d2afdcaf9ec066f1e21781ea5c2c84965b62ab82fdce6b4f` | 156,754 clusters, 19,848 pages, 737.8 MB encoded / 1,208.0 MB decoded, roots 122.4 MB decoded. The largest proxy has 86,338 triangles. |
| `sponza-ivy-split-textures-mac-20261031` | `a27158279aa8dc1433f3ec0a3460e636ac81821429361c80feea597e6e337f03` | 2 BC7 and 1 RGBA8 images, 134,217,780 native bytes. Payloads are identical to `sponza-ivy-textures-material-weights-mac-20261030`. |

The page directories have no `credits_license.txt`, because `build-gltf-geometry.mjs` has no license option; the earlier main, Curtains and Trees page packages lack it too. Every asset set and the composition carry the license. All 17 Ivy UV spans are unknown, so the texture streamer requests the finest Ivy mips.

**UV audits and asset sets.** Each UV audit matches its geometry package:

| UV audit | Primitives | UV spans |
| --- | ---: | --- |
| `sponza-curtains-uv-demand-mac-20261031.json` | 43 | 35 finite, 8 unknown |
| `sponza-trees-uv-demand-mac-20261031.json` | 4 | 4 finite |
| `sponza-ivy-split-uv-demand-mac-20261031.json` | 17 | 17 unknown |

The single-source sets are:

| Asset set | Manifest SHA-256 | Notes |
| --- | --- | --- |
| `sponza-curtains-asset-set-mac-20261031` | `80a6d745f987ade675903ee65afcd391e7786a392ec014d7067ded47eb304dcf` | |
| `sponza-trees-asset-set-mac-20261031` | `948764a0f62bd4f200ee26b703bc19f9ebfda475fdbfd865179b20896994d20f` | 2 omitted primitives |
| `sponza-ivy-asset-set-mac-20261031` | `4342da81f7419b48da6766330bceee20eaf469306fe5dcc7a10b6adfa2da09ff` | |

**Composed Sponza set.** `sponza-composed-asset-set-mac-20261031` (manifest `b0a1c43593993ab11a9310f26304efab277605ed373925d611183b7b78924fd0`) uses format `real-engine-asset-composition` v1 with `frame.componentTransform: "identity"`.

- *Contents.* It binds main (the demo set with its camera path), Curtains, Trees and Ivy: 12,833,689 source triangles, 463 paged primitives, 6 omitted primitives and 87 textures (3.83 GB native).
- *Checks while writing.* `write-asset-set.mjs --compose` re-reads and re-hashes every component set, and its geometry, texture, fine-query, UV-audit and camera-path manifests, before writing.
- *Loading.* `src/gltf-package-composition.ts` loads it. Only CPU-side checks have run: a Node load with fetch stubbed to disk gave 463 batches and entities and 87 texture sources.
- *Not yet done.* It is not wired into `/large-scene.html` and has had no browser or visual check. The add-ons' world bounds lie inside the main building's bounds. That supports a shared scene frame but does not prove it.

**Crack audit of the built pages.** `scripts/audit-geometry-boundaries.mjs` decodes and hash-verifies every page. It then requires each non-terminal group to keep its signed welded boundary (`geometryGroupBoundaryAudit`). With that property, every cut that `geometryCutCoverage` accepts has exactly LOD0's net boundary. On 2026-10-31 it found 0 mismatched groups in every package:

| Package | Paged primitives | Groups checked | Mismatched |
| --- | ---: | ---: | ---: |
| Sponza main | 401 | 6,385 | 0 |
| Curtains | 43 | 3,327 | 0 |
| Trees | 2 | 1,625 | 0 |
| Ivy group 16 | 17 | 8,929 | 0 |
| Ivy group 8 | 17 | 17,544 | 0 |

**Animated Knight.** Blender 5.2.2 LTS (glTF exporter 5.2.40) runs `scripts/convert-fbx-gltf.py`; with `--threads 1` the output is byte-identical across runs. The source extraction is in `sponza-knight-source-mac-20261031`. Output:

| File | SHA-256 |
| --- | --- |
| `sponza-knight-gltf-mac-20261031/manifest.json` | `c07ecdb783c0f7a4dc0c226296992b2652a08083015a80807843d7ccd77767a5` |
| `knight.gltf` | `eb5d05b440846355abaadb8b6306b490bd62a6a98b35ac73787e538986140b8c` |
| `knight.bin` | `644b2d88ab49d52416339cd8a8ab870be665321c537b0ee86285ce48c29f915b` |

- *Sources.*
  - Geometry comes from `Exports/FBX/Knight_USD_002.fbx` (SHA-256 `bac1b6f1…`).
  - Its usdPreviewSurface materials are mapped to glTF metallic-roughness and rebuilt from `Scenes/knight_USD_PREVIEW_SURFACE_ANIM_002.rnd.ma`.
  - The FBX keeps the skin deformers but drops the weights. Weights and bind matrices therefore come from skinCluster332, 333 and 334 in `Scenes/knight_ARNOLD_002.rnd.ma`; their input shapes match the FBX points within 8.6e-9 cm.
  - The animation comes from `Exports/FBX/Knight_Animation_Data_Only_002.fbx`.
- *Counts.*
  - 3 meshes, 8 primitives, 70,266 vertices and 122,252 triangles.
  - 8 double-sided materials, 10 textures and images, 88 nodes.
  - One skin with 84 joints and at most 4 influences per vertex.
  - One animation, `Take 001`: 12.458 s (frames 1–300 at 24 fps), 252 channels. No morph targets.
- *Validation.* `scripts/validate-gltf-skinned-animation.mjs` loads the file in Node with the engine's own importer.
  - Skin, animation rig and deformation import without warnings. Joints use 84 of 1,024 and nodes 88 of 4,096; vertices and indices are far below the per-primitive limits.
  - Rest pose × inverse bind is identity within 1.9e-7. 75 of 84 joint matrices differ between t = 0 and t = 1 s.
  - Against the package's USD vertex cache at frames 1, 25, 150 and 300, sword and shield match within 6.4e-6 m and the body within 27.9 mm, the gap between linear and dual-quaternion skinning. The comparison first applies a constant −74.67° heading, which the glTF does not apply.
- *Limit exceeded.* The 10 images decode to 671,088,640 bytes, 2.5× the default `maxDecodedImageBytes` of `loadGltf` (268,435,456). A caller must raise that option to load the file. No engine limit was changed.
- *Not carried over.* `manifest.notCarriedOver` lists 16 entries with evidence:
  - Dual-quaternion skinning, which is now linear blend.
  - Influences above four: 2,189 vertices pruned, with at most 3.03 mm of displacement.
  - The 365-joint AdvancedSkeleton rig with constraints and IK. Its motion is baked into the joint keys.
  - Hair dynamics, ngSkinTools layers and script nodes.
  - Arnold-only material properties, such as sheen, anisotropy and bump2d.
  - Changed rather than dropped: IOR pinned to 1.5, roughness maps repacked from R to G, n-gons triangulated, centimetres converted to metres.
  - Five extra UV sets and the vertex colours.
  - The USD caches and Looks, the Alembic export, environment and preview images, and FBX custom properties.
- *Not yet done.* The knight has not been rendered on the GPU or placed in a scene. Its texture colour spaces and normal-map convention have not been visually verified.

**Bistro Exterior.** Source: `BistroExterior.fbx`, 119,845,216 bytes, SHA-256 `471991629c7e35a88338c48f3565390ad19cc5d1d7c7e067fc2a60b4f3b5f195`.

- *Import.* Blender 5.2.2 LTS imports it in background mode with factory settings (`blender -b --factory-startup -Y --python scripts/blender/convert-fbx-scene.py -- --input <fbx> --output <new dir>`). The import keeps the authored custom normals, exports tangents and the directional light, and does not search for images. The FBX camera animation is written separately as `camera-path.json`: 2,401 frames at 24 fps (100 s), glTF axes.
- *Material map.* `scripts/apply-fbx-material-map.mjs --convention orca-metal-rough` rewrites the materials explicitly:
  - BaseColor RGB, with alpha as coverage.
  - Specular texture: R occlusion, G roughness, B metalness.
  - Normal maps use the DirectX convention; their green channel is flipped when the textures are built.
  - Emissive RGB.
  - Names ending in `.DoubleSided` become double-sided.
  - A material is OPAQUE when all base-color alpha is 255. Otherwise it is MASK with Falcor's cutoff of 0.5.
  - It removes `KHR_materials_specular`, which Blender writes.
- *Output.* `scene.gltf` (1,930,569 bytes, SHA-256 `a3ef4278af68b35f17706b3b84bd151ad79a9c6e4b9bee7f3b548ab35a9be107`) and `scene.bin` (179,130,372 bytes, SHA-256 `18fac23b3c66ce395ed0383734a4fd5c29848001e3cc92b0a92817bfee2956c3`).
- *Counts.*
  - 1,296 meshes, 1,591 primitives and 1,298 nodes.
  - 2,829,226 imported triangles. The README gives 2,799,704 unique and 2,832,120 instance-expanded; those are publisher counts.
  - 132 materials: 120 OPAQUE and 12 MASK, 6 of them double-sided.
  - 405 images and one directional light.
- *Packages.* The geometry build used `--remove-zero-area`, which removed 19 triangles. It pages 1,402 opaque primitives (2,543,852 triangles). The 189 MASK primitives (285,355 triangles) stay source-routed and are not virtualized. Textures: 273 BC7 images and 132 RGBA8 normal maps.
- *Not carried over.* `conversion.json` records what the engine does not use:
  - `TEXCOORD_1`.
  - The occlusion channel. The engine material model does not read occlusion textures, so the scene loader strips them and reports each one.
  - Two fully cut-out MASK materials: `MASTER_Glass_Dirty_MASKED` and `Paris_LiquorBottle_01_Labels`.

`BistroInterior.fbx` and `BistroInterior_Wine.fbx` are extracted but not converted. The wine scene's additional material parameters are still unmapped.
