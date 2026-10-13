# Real Engine

面向代码与 AI 编程的 WebGPU 游戏引擎。核心方向是可组合的虚拟几何、动态全局光照、材质与形变系统，并通过明确的数据接口集成物理。

## 项目计划

- [架构与接口](docs/architecture.md)
- [研发阶段与交付门槛](docs/roadmap.md)
- [基准测试与验收](docs/benchmarks.md)
- [技术调研与一手来源](docs/research.md)
- [阶段 1–2 交付跟踪](docs/phase-1-2-status.md)

## 当前状态

阶段 1.2 进行中：代码定义的 Morph/蒙皮/wave 算子编译为共享 GPU 程序，主视图与阴影共用当前/前帧形变；动态包围盒、generation 场景句柄、增量帧提交和 Render Graph 已接入实际场景，并连接基础 PBR 和 Rapier 固定步物理。

计划中的 Nanite 类虚拟几何、Lumen 类世界空间 GI 和反射均尚未实现，文档中的性能数值是待验证目标。当前是有界顺序形变编译器，材质仍是基础不透明 PBR。完整边界与后续工作见 [实现状态](docs/implementation.md)。

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

2026-10-14 本地验证：65 项单元测试、TypeScript 与生产构建通过；Chrome 154 的浏览器测试覆盖实际渲染、六骨骼多实例形变、法线、边界、历史、缓存与错误路径。尚未完成性能档位、长期稳定性或跨厂商认证。

## 代码入口

| 文件 | 职责 |
| --- | --- |
| src/contracts.ts | 与游戏内容类别无关的场景输入 |
| src/deformation.ts | 算子编译、验证、参考求值和动态边界 |
| src/scene-store.ts | generation 句柄、版本、增量事件及快照 |
| src/render-graph.ts | pass 读写依赖编译、校验和执行 |
| src/renderer.ts | WebGPU 资源、形变共享、阴影与表面调度 |
| src/shaders/deform.wgsl | 融合算子的当前/前帧执行模板 |
| src/shaders/surface.wgsl | 直接光 PBR、阴影与运动向量 |
| src/physics.ts | Rapier 适配与固定步姿态快照 |
| src/scene.ts | 用代码组合通用能力的测试场景 |
