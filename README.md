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

阶段 2 已建立离线几何簇层级、简化与压缩分页，并接入全部驻留的 GPU LOD、视锥剔除、间接绘制和分簇颜色演示；共享形变的世界空间查询供有界漫反射 GI 使用，另有可选的跨帧多次反弹反馈与实验性 Final Gather。页面流送、遮挡剔除、完整 GI、镜面反射和参与介质仍待实现，路线图中的目标规模与性能门槛仍待验证。当前形变和材质编译器均有明确上限；本轮加入基础法线贴图及 glTF `normalTexture`，仍无多层材质或透射。阶段 2 后期将通过大规模完整场景联合验收画质和性能；现有小模型用于正确性验证及有限场景的性能基线。完整边界与后续工作见 [实现状态](docs/implementation.md)。

此前画质快照提供代码配置的阴影范围/分辨率、单帧 FXAA、共享形变/材质/遮挡的发光面主动采样，以及可选的实验性 Final Gather。旧地面六点远场参考平均误差由约 8.0% 降至 5.6%，但使用 3 cm 射线起点偏移，不能证明接触区正确。在独立 4096 路径、最多 12 段、0.1 mm 起点偏移的接触区参考中，开启 Final Gather 后五个可比较像素的平均 RGB 相对误差由 52.18% 降至 11.29%，最差仍为 30.77%，尚未达到目标的平均 10% / 单项 20% 门槛。1 cm 目标位置对应的像素命中白块，不参与地面 HDR 对比。此前本地 250 项 CPU 与 50 项原生 NVIDIA 浏览器回归是旧快照。本轮法线贴图版本的 `npm run check` 通过 262 项 CPU 测试（34 文件）、TypeScript 与生产构建，原生 NVIDIA `npm run test:browser` 全部 55 项通过，释放后账本归零且无 GPU validation 错误。有限平面联合测试验证了法线贴图、共享形变、世界查询、几何半球和实际减面 LOD；结果不外推到复杂资产。日志及三轮硬件镜头基准的边界见 [实现状态](docs/implementation.md)。

## 本地运行

需要 Node.js 24 与支持 WebGPU 的浏览器。使用 localhost 或 HTTPS。

```sh
npm ci
npm run dev
```

打开终端给出的本地地址。拖动旋转视角，滚轮缩放；侧栏支持暂停、重置、刚体增删、GPU 形变验证与 JSON 诊断导出。`?trees=128` 可调整树木实例数，上限为 256。页面是测试场景与诊断入口，不是编辑器。

`/geometry.html` 展示 GPU 自动 LOD：正常材质、稳定簇颜色和 LOD 层级三种视图，支持误差调整、固定相机轨迹、暂停逐帧、冻结选择视点和按 H 隐藏界面。雕塑的实际簇/三角形数量来自限频 GPU 读回，并标注采样帧。当前模型为 5,120 三角形的程序雕塑，三个实例共享层级；这是正确性与可视化原型，不是最终大规模场景或性能成绩。

## 验证

`/gi.html` 展示世界空间漫反射 GI：红蓝墙面和发光条通过真实场景查询照亮白色物体，可切换单次/跨帧多次反弹（演示默认多次）、开关 GI、改变墙色及太阳参数、暂停逐帧、按 H 隐藏面板。引擎的 `diffuseGi.multiBounce` 默认 `false`，运行时用 `setDiffuseGiMultiBounce(boolean)` 切换，`invalidateDiffuseGiHistory()` 可显式丢弃旧反馈。探针使用 8×8 八面体方向距离矩和低置信度平滑权重；`visibilityWeightThreshold` 默认 0.2、设 0 可关闭额外曲线。固定网格仍每帧全量更新；无镜面反弹、时域平均、自动布点或探针流送。关闭 GI 时没有旧常量环境补光。

`/gi.html?finalGather=1` 额外开启实验性 Final Gather 预览，起始为固定镜头，持续渲染以积累近场样本；常规 `/gi.html` 保持世界探针路径。代码可通过 `DiffuseGiOptions.finalGather` 配置，并用 `setDiffuseGiFinalGatherEnabled(boolean)` 单独启停。普通镜头运动可将前帧原始锚点重投影；切镜头、动态几何或光照/材质突变仍保守地整体重置。此预览的近场误差和帧间隔仍有未通过门槛，见 [实现状态](docs/implementation.md#实验性屏幕-final-gather)。

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

本机 Edge 可显式选择并检查两个硬件适配器；下面是 PowerShell 示例，两次均不使用软件 GPU flags：

```powershell
$env:PLAYWRIGHT_CHANNEL = 'msedge'
$env:PLAYWRIGHT_GPU = 'hardware'
$env:PLAYWRIGHT_EXPECT_GPU_VENDOR = 'intel'
npm run test:browser
$env:PLAYWRIGHT_HIGH_PERFORMANCE_GPU = '1'
$env:PLAYWRIGHT_EXPECT_GPU_VENDOR = 'nvidia'
npm run test:browser
```

`--force-high-performance-gpu` 在本机选择 `nvidia / ampere`，默认 Edge 选择 `intel / gen-12lp`。主机安装 RTX 3070 Laptop，但 WebGPU adapter 不报告具体型号。其他机器应按实际 adapter 调整预期厂商；两个浏览器运行之间清理上述环境变量，避免影响之后的默认测试。

2026-10-14 阴影/GI 修正前的本地 `npm run check` 通过 222 项 CPU 单元测试、TypeScript 与生产构建（`artifacts/local-check-gi-confidence-final-20261014.log`）；Edge 默认 Intel 通过 28 项浏览器功能测试，强制高性能 NVIDIA 的最新 8×8 GI 路径通过 31 项（`artifacts/local-nvidia-gi-confidence-final-20261014.log`）。基础压力场景各完成三次 1800 帧短程硬件基准，固定 1080p 几何演示及旧/新 GI 版本也各完成三次 NVIDIA 短程基准。另一次基础压力场景的 NVIDIA 30 分钟运行通过资源/功能稳定性检查，但 rAF p95 为 34.8 ms，未达到初始 33.3 ms 帧间隔预算。详情见[实现状态](docs/implementation.md#基准运行器与首轮证据)和[验收协议](docs/benchmarks.md#结果与回归)。几何固定镜头全部选中 LOD0，尚未证明减面收益；30 分钟结果也不代表 GI/LOD/雾的联合长测、最终画质或大场景性能通过。

同日后续提交 `00f989a` 已通过 [远程 Linux 软件 GPU 回归](https://github.com/fomalhautb/real-engine/actions/runs/38513178363)：188 项单元测试和 18 项浏览器测试。覆盖共享形变 BVH 世界查询、线性 HDR 与曝光/显示转换、设备丢失，以及变化槽位上传和静止姿态复用；均未在用户使用电脑期间进行本机 GPU 压测。这些功能证据不代表阶段 2 的完整系统或性能目标已完成。

## 可重复基准

安装系统 Chrome 后可执行：

```sh
npm run benchmark -- --frames 1800 --warmup 300 --trees 128 --repeat 3
npm run benchmark -- --seconds 1800 --warmup 300 --trees 128 --stress --repeat 1
```

第一条采集固定轨迹；第二条按真实时间运行 30 分钟，循环增删刚体、换材质、重置、调整分辨率和切换镜头。默认以正常 GPU 配置运行独立的 headless Chrome；`--software` 仅用于功能排错，`--headed` 可显示测试窗口，`--channel msedge` 可选 Edge。运行环境和实际启动参数写入报告。

本机硬件基础压力基准可指定 Edge 与预期厂商：

```sh
npm run benchmark -- --frames 1800 --warmup 300 --trees 128 --stress --repeat 3 --channel msedge --expect-vendor intel --out artifacts/benchmarks/local-foundation
npm run benchmark -- --frames 1800 --warmup 300 --trees 128 --stress --repeat 3 --channel msedge --high-performance-gpu --expect-vendor nvidia --out artifacts/benchmarks/local-foundation-nvidia
```

GI/几何演示使用独立的固定分辨率入口；例如：

```sh
node scripts/benchmark-demo.mjs --page gi --channel msedge --warmup 300 --frames 1800 --repeat 3 --width 1920 --height 1080 --high-performance-gpu --expect-vendor nvidia --out artifacts/benchmarks/local-gi-nvidia
node scripts/benchmark-demo.mjs --page geometry --channel msedge --warmup 300 --frames 1800 --repeat 3 --width 1920 --height 1080 --high-performance-gpu --expect-vendor nvidia --out artifacts/benchmarks/local-geometry-nvidia
```

演示基准在首帧前关闭低频诊断读回、重置相机轨迹，并使用独立生产构建和固定场景；GI 与几何结果不能直接相减解释单个 pass 的代价。预期厂商会对照 Renderer 实际使用的 adapter 检查。此前 8×8 GI 的证据在 `artifacts/benchmarks/local-gi-directional-nvidia-20261014`，旧六方向版本在 `local-gi-nvidia-20261014-v3`，两者源码/构建指纹不同。输出目录必须是 `artifacts/` 下尚不存在的路径。

GI 演示基准可用 `--final-gather --camera orbit` 测量实验路径的确定性慢速镜头运动；`--camera fixed` 为默认，`orbit` 仅用于 `--page gi`。报告逐帧保存实际相机位置。此前 orbit 证据在 `artifacts/benchmarks/local-final-gather-camera-final-20261014`；其最差一次 rAF p95 为 20.9 ms，不能视为 60 FPS 达标。本轮最终共享布局回归在 `artifacts/benchmarks/local-normal-layout-release-camera-20261015`，相同 orbit 与 Final Gather 下三次 GPU pass 总和 p95 为 4.5907968/4.5252608/4.718592 ms，rAF p95 为 7.0/7.0/13.9 ms；该 GI 房间未启用 `normalTexture`，不能据此推断复杂法线贴图成本或跨构建加速。

结果保存在 `artifacts/benchmarks/`，包含源码/生产构建哈希、环境清单、逐帧 CSV、统计 JSON、截图和运行期间的内存观察。生产构建复制到独立目录，后续开发不会改变正在测量的版本。rAF 间隔表示浏览器回调节奏，GPU 时间是稀疏异步样本；两者均不等于显示器实际呈现时间。基础压力场景的长测证据在 `artifacts/benchmarks/local-stability-nvidia-20261014`，测得 1,800,000.2 ms、167,349 帧、92 个完整事件循环、无应用 GPU 账本增长或错误且释放归零；rAF p95 34.8 ms、p99 62.5 ms，10,401 帧（6.215%）超过初始预算。压力序列切换 1080p/810p，运行中有少量短 CPU 测试；这不代表固定 1080p、严格无干扰性能认证或大型 GI/LOD/镜面反射/体积雾联合长测。完整协议与边界见 [实现状态](docs/implementation.md#基准运行器与首轮证据)。

## 离线几何构建

```sh
npm run build:geometry -- --input path/to/static.obj --output artifacts/geometry/example --coverage opaque
```

输入为静态 OBJ，必须显式确认使用 Opaque 覆盖；工具保留对象/材质分段和来源哈希，不转换 MTL。输出包含简化簇层级、可独立解码的压缩页面、根页面映射与构建参数，拒绝覆盖已有输出目录。层级 JSON 用于离线审计，不是运行时驻留格式。GPU LOD 目前直接使用内存中的层级，压缩页面尚未接入运行时流送。支持范围见 [离线构建说明](docs/implementation.md#离线几何构建与页面)。

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

材质定义可选 `normal: Vec4Expr` 提供 RGB 编码的切线空间法线，`normalScale: ScalarExpr` 缩放其 XY 分量（允许负值）；未配置时沿用原法线。glTF `normalTexture` 占线性纹理槽 3，`scale` 放入参数槽 3 的 x 分量。缺失有效 authored `TANGENT` 时，消费端按已变形三角形的 UV 导数回退；尚未生成 MikkTSpace 切线，不能保证完整 glTF 法线贴图保真。详细边界与验证见 [实现状态](docs/implementation.md)。

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
| src/geometry-lod-data.ts、src/gpu-geometry-lod.ts | 层级验证/打包、GPU 选簇、索引压紧和间接绘制参数 |
| src/geometry-demo.ts、src/geometry-demo-scene.ts | 实际簇分色、LOD 层级及可重复镜头演示 |
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
