# Third-party glTF test assets

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
