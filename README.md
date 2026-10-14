# Real Engine

面向代码与 AI 编程的 WebGPU 游戏引擎。核心方向是可组合的虚拟几何、动态全局光照、材质与形变系统，并通过明确的数据接口集成物理。

## 项目计划

- [架构与接口](docs/architecture.md)
- [研发阶段与交付门槛](docs/roadmap.md)
- [基准测试与验收](docs/benchmarks.md)
- [技术调研与一手来源](docs/research.md)
- [阶段 1–2 交付跟踪](docs/phase-1-2-status.md)
- [第三方测试资产与许可证](docs/assets.md)

## 当前状态

阶段 1.2 进行中：代码定义的 Morph/蒙皮/wave 算子编译为共享 GPU 程序，主视图与阴影共用当前/前帧形变；动态包围盒、generation 场景句柄、增量帧提交和 Render Graph 已接入实际场景。现有代码生成材质表达式支持 PBR/Unlit、Opaque/Masked、双面、纹理与共享阴影覆盖；glTF/GLB 蒙皮与 Morph 动画、OBJ 卡片、纹理地面及 Rapier 固定步刚体同场运行。

阶段 2 已开始建立离线几何簇层级、简化与压缩分页，以及共享形变的世界空间查询基础。运行时 GPU LOD/流送、世界空间 GI、反射和参与介质仍待实现，文档中的性能数值是待验证目标。当前形变和材质编译器均有明确上限；没有法线贴图、多层材质或透射。阶段 2 后期将通过大规模完整场景联合验收画质和性能；现有小模型仅用于正确性检查。完整边界与后续工作见 [实现状态](docs/implementation.md)。

## 本地运行

需要 Node.js 24 与支持 WebGPU 的浏览器。使用 localhost 或 HTTPS。

```sh
npm ci
npm run dev
```

打开终端给出的本地地址。拖动旋转视角，滚轮缩放；侧栏支持暂停、重置、刚体增删、GPU 形变验证与 JSON 诊断导出。`?trees=128` 可调整树木实例数，上限为 256。页面是测试场景与诊断入口，不是编辑器。

## 验证

```sh
npm run check
npx playwright install chromium
npm run test:browser
```

默认浏览器测试使用 SwiftShader，验证功能而非真实硬件帧率。

Linux 无显示器环境先用 `npx playwright install --with-deps chromium` 安装依赖，再执行 `xvfb-run -a npm run test:browser`，保证 WebGPU canvas 的合成与截图路径可用。

安装系统 Chrome 的 Windows 环境可运行：

```powershell
$env:PLAYWRIGHT_CHANNEL = 'chrome'
npm run test:browser
$env:PLAYWRIGHT_GPU = 'hardware'
npm run test:browser
```

第二次运行不启用实验 GPU flags，使用正常硬件适配器。没有兼容 GPU 时测试明确失败，不跳过验证。测试截图和诊断附在 Playwright 输出中，不提交生成文件。

2026-10-14 本地验证：166 项单元测试、TypeScript 与生产构建通过；Chrome 154 正常硬件配置通过 12 项浏览器测试，覆盖 glTF 动画、仿射骨骼、材质覆盖、Compute 材质数值、独立采样器、物理、运动向量、历史、预算、基准时长与错误路径。这些是功能证据，尚未完成性能档位、长期稳定性或跨厂商认证。

同日后续提交 `00f989a` 已通过 [远程 Linux 软件 GPU 回归](https://github.com/fomalhautb/real-engine/actions/runs/38513178363)：188 项单元测试和 18 项浏览器测试。覆盖共享形变 BVH 世界查询、线性 HDR 与曝光/显示转换、设备丢失，以及变化槽位上传和静止姿态复用；均未在用户使用电脑期间进行本机 GPU 压测。这些功能证据不代表阶段 2 的完整系统或性能目标已完成。

## 可重复基准

安装系统 Chrome 后可执行：

```sh
npm run benchmark -- --frames 1800 --warmup 300 --trees 128 --repeat 3
npm run benchmark -- --seconds 1800 --warmup 300 --trees 128 --stress --repeat 1
```

第一条采集固定轨迹；第二条按真实时间运行 30 分钟，循环增删刚体、换材质、重置、调整分辨率和切换镜头。默认以正常 GPU 配置运行独立的 headless Chrome；`--software` 仅用于功能排错，`--headed` 可显示测试窗口，`--channel msedge` 可选 Edge。运行环境和实际启动参数写入报告。

结果保存在 `artifacts/benchmarks/`，包含源码/生产构建哈希、环境清单、逐帧 CSV、统计 JSON、截图和运行期间的内存观察。生产构建复制到独立目录，后续开发不会改变正在测量的版本。rAF 间隔表示浏览器回调节奏，GPU 时间是稀疏异步样本；两者均不等于显示器实际呈现时间。当前仅测基础场景，不代表大型场景、GI、反射或体积雾性能。完整协议与边界见 [实现状态](docs/implementation.md#基准运行器与首轮证据)。

## 离线几何构建

```sh
npm run build:geometry -- --input path/to/static.obj --output artifacts/geometry/example --coverage opaque
```

输入为静态 OBJ，必须显式确认使用 Opaque 覆盖；工具保留对象/材质分段和来源哈希，不转换 MTL。输出包含简化簇层级、可独立解码的压缩页面、根页面映射与构建参数，拒绝覆盖已有输出目录。层级 JSON 用于离线审计，不是运行时驻留格式。当前页面尚未接入 Renderer；自动 LOD、实际簇分色与录屏演示按阶段 2 后续交付。支持范围见 [离线构建说明](docs/implementation.md#离线几何构建与页面)。

## 材质接口示例

下面的 `mesh`、`instance`、`canvas` 和 `texture` 由应用提供；`texture` 是 RGBA8 `TextureAsset`。`SceneBatch.material` 在创建 Renderer 时提供初始材质；运行时只改参数不会重新编译 shader。

```ts
import { m, type MaterialDefinition } from './src/material';
import type { SceneBatch } from './src/contracts';
import { Renderer } from './src/renderer';

const definition: MaterialDefinition = {
  shading: 'unlit', alphaMode: 'mask', doubleSided: true,
  baseColor: m.multiply(m.multiply(m.parameter(0), m.instanceColor), m.texture(0)),
  metallic: 0, roughness: 1, emissive: m.constant([0, 0, 0, 0]), alphaCutoff: 0.5,
};
const batch: SceneBatch = {
  name: 'card', mesh, capacity: 1, instances: [instance],
  deformation: { jointCount: 4, nodes: [] }, morphTargets: [],
  material: { definition, parameters: new Float32Array([1, 1, 1, 1]), textures: [texture] },
};
const renderer = await Renderer.create(canvas, [batch]);
renderer.setMaterialParameters('card', new Float32Array([0.8, 1, 1, 1]));
```

`await renderer.setMaterial('card', binding)` 异步准备后原子替换整套材质；`renderer.setSun({ direction, color })` 更新太阳方向与线性 RGB 辐亮度。详见 [实现状态](docs/implementation.md)。

## glTF 动画接口示例

`canvas` 由应用提供，时间单位为秒。实例共享不可变资产，各自采样姿态；所有实例的实体 ID 和批次名称需唯一。外部文件必须支持浏览器跨域访问或由本地托管。

```ts
import { loadGltf } from './src/gltf-load';
import { GltfSceneInstance } from './src/gltf-scene';
import { Renderer } from './src/renderer';

const asset = await loadGltf('/assets/gltf/SimpleSkin/glTF/SimpleSkin.gltf');
const actor = new GltfSceneInstance(asset, { name: 'actor', firstEntityId: 1 });
const renderer = await Renderer.create(canvas, actor.batches);
actor.sample(1, 0, true); // 时间、动画索引、循环；第四参数 true 用于跳播时使历史失效。
renderer.render([4, 3, 6], [0, 1, 0]);
// 应用下一帧再次 sample/render；退出时 renderer.dispose()。
```

## 代码入口

| 文件 | 职责 |
| --- | --- |
| src/contracts.ts | 与游戏内容类别无关的场景输入 |
| src/deformation.ts | 算子编译、验证、参考求值和动态边界 |
| src/scene-store.ts | generation 句柄、版本、增量事件及快照 |
| src/render-graph.ts | pass 读写依赖编译、校验和执行 |
| src/renderer.ts | WebGPU 资源、形变共享、阴影与表面调度 |
| src/material.ts、src/gpu-material.ts | 有界材质表达式、覆盖程序、GPU 材质与纹理资源 |
| src/mesh-import.ts、src/texture-assets.ts | OBJ 网格导入与 RGBA8 纹理/mip 处理 |
| src/geometry-hierarchy.ts、src/geometry-pages.ts | 离线简化簇 DAG、CPU 参考 LOD 与有界压缩页面 |
| scripts/build-geometry.mjs | 静态 Opaque OBJ 构建、来源/页面哈希与原子输出 |
| src/gltf-binary.ts、src/gltf-import.ts、src/gltf-load.ts | 有界 glTF/GLB 解码、网格/蒙皮/材质导入与资源加载 |
| src/animation.ts、src/gltf-scene.ts | TRS/Morph 动画采样、层级姿态和统一场景输入 |
| src/shaders/deform.wgsl | 融合算子的当前/前帧执行模板 |
| src/shaders/surface.wgsl | 直接光 PBR、阴影与运动向量 |
| src/shaders/resolve.wgsl | 线性 HDR 的曝光、色调映射与 sRGB 输出 |
| src/gpu-world-query.ts | 共享形变 BVH refit 与有界世界射线命中 |
| src/participating-media.ts | 均匀介质积分与合成的 CPU 数值参考 |
| src/physics.ts | Rapier 适配与固定步姿态快照 |
| src/scene.ts | 用代码组合通用能力的测试场景 |
| src/benchmark.ts、src/benchmark-scenario.ts、src/benchmark-stats.ts | 固定轨迹/事件、真实时长运行与有界逐帧统计 |
| scripts/benchmark.mjs | 独立生产构建、浏览器运行、环境/构建指纹与结果导出 |
