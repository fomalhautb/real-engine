# Real Engine

面向代码与 AI 编程的 WebGPU 游戏引擎。核心方向是可组合的虚拟几何、动态全局光照、材质与形变系统，并通过明确的数据接口集成物理。

## 项目计划

- [架构与接口](docs/architecture.md)
- [研发阶段与交付门槛](docs/roadmap.md)
- [基准测试与验收](docs/benchmarks.md)
- [技术调研与一手来源](docs/research.md)
- [阶段 1–2 交付跟踪](docs/phase-1-2-status.md)

## 当前状态

阶段 1.2 进行中：代码定义的 Morph/蒙皮/wave 算子编译为共享 GPU 程序，主视图与阴影共用当前/前帧形变；动态包围盒、generation 场景句柄、增量帧提交和 Render Graph 已接入实际场景。现有代码生成材质表达式支持 PBR/Unlit、Opaque/Masked、双面、纹理与共享阴影覆盖；OBJ 导入的原创卡片和重复纹理地面与 Rapier 固定步刚体同场运行。

计划中的 Nanite 类虚拟几何、Lumen 类世界空间 GI 和反射均尚未实现，文档中的性能数值是待验证目标。当前形变和材质编译器均有明确上限；没有 glTF/蒙皮资产导入、法线贴图、多层材质或透射。完整边界与后续工作见 [实现状态](docs/implementation.md)。

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

2026-10-14 本地验证：107 项单元测试、TypeScript 与生产构建通过；Chrome 154 的 SwiftShader 和正常硬件配置各通过 8 项浏览器测试，覆盖形变、材质覆盖、Compute 材质数值、纹理、物理、历史、预算与错误路径。这些是功能证据，尚未完成性能档位、长期稳定性或跨厂商认证。

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
| src/shaders/deform.wgsl | 融合算子的当前/前帧执行模板 |
| src/shaders/surface.wgsl | 直接光 PBR、阴影与运动向量 |
| src/physics.ts | Rapier 适配与固定步姿态快照 |
| src/scene.ts | 用代码组合通用能力的测试场景 |
