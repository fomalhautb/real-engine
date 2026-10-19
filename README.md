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

阶段 1.2 进行中：代码定义的 Morph/蒙皮/wave 算子编译为共享 GPU 程序，主视图与阴影共用当前/前帧形变；动态包围盒、generation 场景句柄、增量帧提交和 Render Graph 已接入实际场景。现有代码生成材质表达式支持 PBR/Unlit、Opaque/Masked、双面、纹理与共享阴影覆盖；glTF/GLB 蒙皮与 Morph 动画、缺失切线时的按需生成、OBJ 卡片、纹理地面及 Rapier 固定步刚体同场运行。

阶段 2 已建立离线几何簇层级、简化与压缩分页，并接入 GPU LOD、视锥剔除、间接绘制和分簇颜色演示；共享形变的世界空间查询供有界漫反射 GI 使用，另有可选的跨帧多次反弹反馈与实验性 Final Gather。物理 GPU 页池现已接入主 Renderer 的静态 Opaque 表面：GPU 根据实际驻留状态整组选簇并间接绘制，独立组需求驱动预算内的整组准入、祖先保留和有界下载/上传。全部流送批次共用有界 Worker 解码池，哈希、解压和载荷检查在后台执行；取消覆盖下载与解码。分页表面复用材质、法线贴图和当前/前帧变换语义。新增 `RenderBatch` 的源网格可省略 `StreamedSceneBatch`：固定有界查询代理供阴影、世界查询和 GI 使用，分页表面仍可绘制更细的相机 LOD；旧的完整 `mesh` 流送路径仍可用。当前只支持静态 Opaque，源网格可省略批次切换到 Masked 会显式拒绝，实例仍可仿射运动。代理命中具有独立的三角形编号与拓扑版本，不能当成高模表面的精确命中；两阶段主视图遮挡剔除已可选接入，完整 GI 仍待交付。新增单段世界空间反射，复用当前材质、形变、太阳遮挡及可选 GI 探针；支持受控光滑/粗糙反射，仍无反射时序积累或镜中镜递归。主视图已有有界 GPU 体积雾：全局高度介质、局部盒/球体积、共享太阳阴影与线性 HDR 合成；新增默认关闭的可选世界空间时序重建，提供有界历史与显式重置。反射光路已借用同一世界介质定义进行有界积分；点/聚光灯、空气中的低频 GI 及透明路径的介质积分仍未实现，时序重建也未完成遮挡体局部历史拒绝或最终画质验收。路线图中的目标规模与性能门槛仍待验证。当前形变和材质编译器均有明确上限；现有基础法线贴图及 glTF `normalTexture`，仍无多层材质或透射。阶段 2 后期将通过大规模完整场景联合验收画质和性能；现有小模型用于正确性验证及有限场景的性能基线。完整边界与后续工作见 [实现状态](docs/implementation.md)。

`RendererOptions.occlusion` 可为静态 Opaque 的驻留/分页 GPU LOD 开启两阶段遮挡：前帧深度只暂缓预测不可见簇，本帧首遍深度决定最终剔除并同帧补绘恢复簇。普通、Masked 和共享形变对象仍完整绘制并可遮挡，阴影/世界查询/GI/反射不按相机遮挡裁剪；预取页请求仍按视锥与误差发出，冻结选择视点时绕过遮挡。默认 2 像素采样、`depthBias: 0.00002`，64 MiB 预算覆盖两份 `r32float` 金字塔与 208 字节参数，resize 先预检。`gpuMs.occlusion` 计入建图和恢复选簇，补绘计入 surface；本轮完整 NVIDIA 功能回归已通过，性能及大场景验收仍待完成。详见[两阶段遮挡剔除](docs/implementation.md#可选两阶段遮挡剔除)。

此前画质快照提供代码配置的阴影范围/分辨率、单帧 FXAA、共享形变/材质/遮挡的发光面主动采样，以及可选的实验性 Final Gather。旧地面六点远场参考平均误差由约 8.0% 降至 5.6%，但使用 3 cm 射线起点偏移，不能证明接触区正确。在独立 4096 路径、最多 12 段、0.1 mm 起点偏移的接触区参考中，开启 Final Gather 后五个可比较像素的平均 RGB 相对误差由 52.18% 降至 11.29%，最差仍为 30.77%，尚未达到目标的平均 10% / 单项 20% 门槛。1 cm 目标位置对应的像素命中白块，不参与地面 HDR 对比。此前本地 250 项 CPU 与 50 项原生 NVIDIA 浏览器回归，以及法线贴图版本的 262 项 CPU（34 文件）与 55 项原生 NVIDIA 浏览器回归，均为历史快照。此前 glTF 切线生成版本的 `npm run check` 通过 280 项 CPU 测试（36 文件）、TypeScript 与生产构建；原生 Edge `nvidia / ampere` 非 fallback 浏览器回归 56 项全部通过，GPU validation 错误为空且释放后应用账本归零。实际 glTF 的 GPU 联合用例覆盖生成切线、Morph/仿射蒙皮、法线贴图 PBR、世界查询和几何法线；位置/法线/切线误差小于 `3e-4`，边界越界为零。证据见 `artifacts/mikk-tangents-check-20261015.log` 与 `artifacts/mikk-tangents-browser-native-20261015.log`。此结果只覆盖测试资产；此前三轮硬件镜头基准未启用该 glTF 生成路径，不能证明复杂资产导入成本或大场景性能。完整边界见 [实现状态](docs/implementation.md)。

此前共享 Worker 页解码通过 `npm run check`（365 项 CPU、47 文件、类型检查与构建）及完整原生 NVIDIA 浏览器回归（65 项）。两个流送批次共用两个 Worker 完成 42 次页解码，既有材质、运动、阴影、查询、GI 和小池换页回归保持通过；取消、设备丢失和初始化失败释放在途工作。生产打包 Worker 的转移/取消/恢复另有定向验证。Worker 提交 `86358ff` 的[远程 CI run 39227014810](https://github.com/fomalhautb/real-engine/actions/runs/39227014810) 已成功；范围、日志与限制见 [实现状态](docs/implementation.md#共享-worker-页解码)。上述结果仍是功能验证，尚未测量大资产吞吐或整帧性能收益。

固定代理版本的完整 `npm run check` 通过 371 项 CPU 测试（48 文件）、TypeScript 与生产构建；原生 NVIDIA 完整浏览器回归 67/67 通过，两个新增源网格可省略用例也在非 fallback Intel 核显定向通过。主视图分页 LOD0、阴影、查询、GI、Final Gather 和小池换页的有限数值联测及账本证据见[固定查询代理](docs/implementation.md#固定查询代理)。Intel 结果不是完整回归或性能成绩；固定代理提交 `789279c` 的[远程 CI run 39230781801](https://github.com/fomalhautb/real-engine/actions/runs/39230781801) 已成功。

世界空间查询默认对容量至少 8 的批次启用实例 BVH；注册至少 8 个批次时，还用场景 BVH 为每条射线筛选候选批次。物体移动或形变后，GPU 从共享三角形结果更新边界；静止批次复用缓存。`worldQueries.instanceAcceleration` 和 `sceneAcceleration` 各自可设为 `'linear'` 保留原路径供对照。材质仍逐批次 dispatch，工作量减少不等于已经证明整帧提速；具体预算与验收见[世界空间查询基础](docs/implementation.md#世界空间查询基础)。

`RendererOptions.maxBatches` 为整个场景设置 primitive 批次预算（默认 256，硬上限 4096），包括普通和分页几何；超过预算在申请 GPU 前报错。世界查询默认继承此容量，直接使用 `GpuWorldQuery` 时可设 `WorldQueryOptions.maxBatches`。场景树与每条射线的候选位图按该容量分配并计入查询 `maxBytes`；显式较低的查询容量不会静默遗漏场景对象。这是容量接口，不是该规模的帧率保证。

实例加速版本通过 378 项 CPU 测试、类型检查与构建、69 项原生 NVIDIA 完整浏览器回归及 4 项 Intel 定向测试。258 个共享形变实例的 11 条射线，工作单位从 5,685 降至 281，命中、画面与 GI 保持一致；此计数不是帧率提升比例。跨批次 BVH 提交 `d7c94d8` 的[远程 CI run 39267473217](https://github.com/fomalhautb/real-engine/actions/runs/39267473217) 已成功，包含 CPU 和 Linux 浏览器作业；旧提交 `41afdbc` 的失败记录仍保留在[实现状态](docs/implementation.md)。

体积雾版本的完整本地 `npm run check` 通过 391 项 CPU 测试（51 文件）、类型检查及生产构建，日志见 `artifacts/fog-check-release-20261019-final.log`。NVIDIA 定向 GPU 数值用例已覆盖解析吸收、零消光发光、HG 太阳散射、局部重叠与高度衰减、共享阴影及形变遮挡后的恢复，另验证 GI 探针未被雾改写和资源释放归零；原生 NVIDIA 完整浏览器回归 74/74 通过，日志见 `artifacts/fog-browser-release-native-20261019-final.log`。Intel 定向 3/3 与 SwiftShader 定向 3/3 也已通过，日志分别见 `artifacts/fog-target-intel-20261019.log`、`artifacts/fog-target-software-20261019.log`；硬件性能验收尚待完成。这些是有限功能证据，不表示阶段 2 或最终画质已通过。

本轮时序版本的完整 `npm run check` 通过 399 项 CPU 测试（51 文件）、类型检查与生产构建，日志 `artifacts/fog-temporal-check-release-20261019.log`；原生 NVIDIA 完整浏览器回归 78/78 通过，日志 `artifacts/fog-temporal-browser-release-native-20261019.log`；Intel 定向 7/7 通过，日志 `artifacts/fog-temporal-target-intel-20261019.log`。SwiftShader 定向 7/7 也已通过，日志 `artifacts/fog-temporal-target-software-20261019.log`。此前 391/74 数字属于无时序版本；提交 `ab53842` 的[远程 CI run 39273105318](https://github.com/fomalhautb/real-engine/actions/runs/39273105318) 已完成且全部作业成功，不代表本轮时序版本的远程结果。

## 本地运行

`RendererOptions.textureStreaming` 可按相机与 UV 密度，在统一图像预算内自动加载、上传和淘汰原生 mip。材质引用来源的低清 `fallback` 身份，之后的新材质也自动使用当前驻留；未完成的图像不会发布，离屏查询保持有效材质。输入、HTTP provider、逐帧/CPU 预算、重试及已验证范围见[自动纹理流送](docs/implementation.md#按视图需求的自动纹理流送)。当前提供整图 mip 尾链调度，尚未完成细分纹理页或最终性能验收。

`npm run build:texture -- --input image.dds --output new-package` 将完整 DX10 RGBA8/BC7 mip 链打包为独立层文件与哈希清单；PNG 可显式配置外部 texconv、格式和色彩空间。浏览器调用 `loadTextureStreamSource('/new-package/manifest.json')` 取得已校验的低清来源，再将它交给自动流送。真实 Sponza 72 张贴图的候选包已构建，石墙材质已有实际浏览器预览；法线压缩质量与整个场景仍需验收。命令、约束和证据见[纹理包构建](docs/implementation.md#纹理包构建清单加载与真实-sponza-贴图)。

需要 Node.js 24 与支持 WebGPU 的浏览器。使用 localhost 或 HTTPS。

```sh
npm ci
npm run dev
```

打开终端给出的本地地址。拖动旋转视角，滚轮缩放；侧栏支持暂停、重置、刚体增删、GPU 形变验证与 JSON 诊断导出。`?trees=128` 可调整树木实例数，上限为 256。页面是测试场景与诊断入口，不是编辑器。

`/geometry.html` 展示 GPU 自动 LOD：正常材质、稳定簇颜色和 LOD 层级三种视图，支持误差调整、固定相机轨迹、暂停逐帧、冻结选择视点和按 H 隐藏界面。雕塑的实际簇/三角形数量来自限频 GPU 读回，并标注采样帧。当前模型为 5,120 三角形的程序雕塑，三个实例共享层级；这是正确性与可视化原型，不是最终大规模场景或性能成绩。

## 验证

`/reflection.html` 展示世界空间反射：镜子中的彩色实体位于相机身后，可切换反射、粗糙度、薄雾、暂停和逐帧。代码通过 `RendererOptions.reflections` 配置采样/距离/内存预算，要求同时配置 `worldQueries`；`setReflectionsEnabled` 切换，`probeReflections` 显式读回有限锚点。反射命中复用材质、Mask 与共享形变，配置 GI 时借用世界探针，配置雾时沿实际反射方向积分世界介质。当前仅单段反射，有限采样与空间重建仍有画质限制，不代表阶段 2 完成，详见[世界空间反射](docs/implementation.md#世界空间反射)。

`/gi.html` 展示世界空间漫反射 GI：红蓝墙面和发光条通过真实场景查询照亮白色物体，可切换单次/跨帧多次反弹（演示默认多次）、开关 GI、改变墙色及太阳参数、暂停逐帧、按 H 隐藏面板。引擎的 `diffuseGi.multiBounce` 默认 `false`，运行时用 `setDiffuseGiMultiBounce(boolean)` 切换，`invalidateDiffuseGiHistory()` 可显式丢弃旧反馈。探针使用 8×8 八面体方向距离矩和低置信度平滑权重；`visibilityWeightThreshold` 默认 0.2、设 0 可关闭额外曲线。固定网格仍每帧全量更新；无镜面反弹、时域平均、自动布点或探针流送。关闭 GI 时没有旧常量环境补光。

`/gi.html?finalGather=1` 额外开启实验性 Final Gather 预览，起始为固定镜头，持续渲染以积累近场样本；常规 `/gi.html` 保持世界探针路径。代码可通过 `DiffuseGiOptions.finalGather` 配置，并用 `setDiffuseGiFinalGatherEnabled(boolean)` 单独启停。普通镜头运动可将前帧原始锚点重投影；切镜头、动态几何或光照/材质突变仍保守地整体重置。此预览的近场误差和帧间隔仍有未通过门槛，见 [实现状态](docs/implementation.md#实验性屏幕-final-gather)。

`/fog.html` 展示柱廊中的局部盒状雾，可切换雾、密度、太阳和镜头轨迹，查看网格与资源诊断。此前未启用时序重建的 NVIDIA 截图为[雾开](artifacts/fog-demo-native-20261019-v2/fog-demo-fog-colonnade-exp-ca6cc-trols-and-bounded-resources/fog-colonnade-on.png) / [雾关](artifacts/fog-demo-native-20261019-v2/fog-demo-fog-colonnade-exp-ca6cc-trols-and-bounded-resources/fog-colonnade-off.png)；本轮可选时序版的实际 NVIDIA 截图为[雾开](artifacts/fog-temporal-demo-native-20261019/fog-demo-fog-colonnade-exp-ca6cc-trols-and-bounded-resources/fog-colonnade-on.png) / [关闭时序、每格中心采样](artifacts/fog-temporal-demo-native-20261019/fog-demo-fog-colonnade-exp-ca6cc-trols-and-bounded-resources/fog-colonnade-centered.png)。演示显式配置 `pixelSize=8`、64 个切片、最多 16 帧历史和 128 MiB 雾预算；截图是有限场景的功能证据，不代表最终画质或性能。

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

GI 演示基准可用 `--final-gather --camera orbit` 测量实验路径的确定性慢速镜头运动；`--camera fixed` 为默认，`orbit` 仅用于 `--page gi`。报告逐帧保存实际相机位置。此前 orbit 证据在 `artifacts/benchmarks/local-final-gather-camera-final-20261014`；其最差一次 rAF p95 为 20.9 ms，不能视为 60 FPS 达标。此前共享布局回归在 `artifacts/benchmarks/local-normal-layout-release-camera-20261015`，相同 orbit 与 Final Gather 下三次 GPU pass 总和 p95 为 4.5907968/4.5252608/4.718592 ms，rAF p95 为 7.0/7.0/13.9 ms；该 GI 房间未启用 `normalTexture` 或 glTF 切线生成，不能据此推断新导入成本、复杂法线贴图成本或跨构建加速。

结果保存在 `artifacts/benchmarks/`，包含源码/生产构建哈希、环境清单、逐帧 CSV、统计 JSON、截图和运行期间的内存观察。生产构建复制到独立目录，后续开发不会改变正在测量的版本。rAF 间隔表示浏览器回调节奏，GPU 时间是稀疏异步样本；两者均不等于显示器实际呈现时间。基础压力场景的长测证据在 `artifacts/benchmarks/local-stability-nvidia-20261014`，测得 1,800,000.2 ms、167,349 帧、92 个完整事件循环、无应用 GPU 账本增长或错误且释放归零；rAF p95 34.8 ms、p99 62.5 ms，10,401 帧（6.215%）超过初始预算。压力序列切换 1080p/810p，运行中有少量短 CPU 测试；这不代表固定 1080p、严格无干扰性能认证或大型 GI/LOD/镜面反射/体积雾联合长测。完整协议与边界见 [实现状态](docs/implementation.md#基准运行器与首轮证据)。

## 离线几何构建

```sh
npm run build:geometry -- --input path/to/static.obj --output artifacts/geometry/example --coverage opaque
```

输入为静态 OBJ，必须显式确认使用 Opaque 覆盖；工具保留对象/材质分段和来源哈希，不转换 MTL。输出包含简化簇层级、可独立解码的压缩页面、根页面映射、`query-proxy.bin` 与构建参数，拒绝覆盖已有输出目录。可选 `--query-error` 使用局部空间误差阈值，默认 `1e30` 选根切分，设 `0` 选 LOD0。层级 JSON 用于离线审计，不是运行时驻留格式。原 `geometryLod` 路径直接使用内存层级；`geometryStream` 可使用完整源 `mesh`，也可通过 `StreamedSceneBatch` 省略它并提供固定代理供阴影/查询/GI 使用。代理默认上限为 65,536 三角形、262,144 顶点和 32 MiB 编码文件大小；详见 [离线构建说明](docs/implementation.md#离线几何构建与页面)及[固定查询代理](docs/implementation.md#固定查询代理)。

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

材质定义可选 `normal: Vec4Expr` 提供 RGB 编码的切线空间法线，`normalScale: ScalarExpr` 缩放其 XY 分量（允许负值）；未配置时沿用原法线。glTF `normalTexture` 占线性纹理槽 3，`scale` 放入参数槽 3 的 x 分量。导入带 `normalTexture` 且缺失有效 `NORMAL`＋`TANGENT` 的 glTF primitive 时，加载器按需准备 WASM，导入器生成 MikkTSpace-compatible 基础和 Morph 目标切线，必要时拆分镜像 UV 接缝并重映射顶点流；同步调用 `importGltf` 前可显式 `await prepareGltfTangents()`。法线贴图需要 `TEXCOORD_0`。代码构建的网格未提供切线时仍可使用已变形三角形的 UV 导数回退。生成结果不保证与任意烘焙器逐位相同，复杂资产保真仍需验证。详细边界与验证见 [实现状态](docs/implementation.md)。

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
| src/gltf-binary.ts、src/gltf-import.ts、src/gltf-tangents.ts、src/gltf-load.ts | 有界 glTF/GLB 解码、网格/蒙皮/材质导入、按需切线生成与资源加载 |
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
