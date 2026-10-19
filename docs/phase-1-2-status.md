# 阶段 1–2 的完整交付跟踪

目标：完成 roadmap.md 中整个阶段 1 和阶段 2。本文跟踪证据，不替代或缩减原路线及 benchmarks.md 的验收条件。状态更新于 2026-10-19；本轮接入主 Renderer 共用的 Worker 页解码池，限制共享在途字节和工位，将取消传到下载与解码；此前整组预算流送保持有效。阴影、查询和 Masked 回退仍保留完整源网格，完整流送与大场景验收继续进行。远程 CI 按各提交分别记录。

## 阶段 1

下述 250 项 CPU、50 项浏览器及早期性能数值来自法线贴图加入前的历史快照。法线贴图、glTF 切线生成、CPU 流送、物理页池及主 Renderer 接入的证据分别记录于 implementation.md；此前提交 `20e3508c`、`0267cbe1`、`8fdf9b61`、`e7214d3`、`c5f6cfb` 的远程 CI 均已成功。本轮 Worker 页解码通过 `npm run check` 的 365 项 CPU 测试（47 文件）、TypeScript 与生产构建，以及原生 NVIDIA 的 65 项完整浏览器回归。尚未测量目标规模的流送性能或大资产导入成本。

| 要求 | 当前证据 | 剩余工作 |
| --- | --- | --- |
| WebGPU 运行时与可诊断错误 | renderer/main；真实浏览器初始化、渲染、resize、释放及失败路径测试；远程软件 GPU 验证显式设备销毁后报错、停帧、查询拒绝和释放 | 扩大浏览器/厂商矩阵；真实驱动丢失及恢复路径 |
| 场景身份、generation、版本与增量变更 | SceneStore；DemoScene.commitFrame；快照隔离、增删复用和版本测试；异步原子换材质及参数更新接口 | 换材质与场景事件的更广联合回归、资源身份/失效语义 |
| 代码形变图、当前/前帧与边界 | 有界顺序算子编译器；Morph/skin/wave 融合 WGSL；glTF 2.0/GLB、蒙皮与 Morph 动画；刚性/仿射骨骼法线；authored 或导入生成的切线随共享 Morph → skin → wave 与实例 handedness 变换，代码网格仍可选 UV 导数回退；glTF 按需生成 MikkTSpace-compatible 基础/目标切线，联合接缝拆分并重映射蒙皮和 Morph；法线 fixture 各实例 GPU 位置/法线/切线误差小于 `3e-4`、边界越界为零；静止批次缓存、按变化槽位上传、无动画 glTF 姿态缓存 | GPU 按脏实例调度、规模性能与算子组合的更广回归；复杂烘焙资产保真、Worker 导入和大资产开销仍需验收 |
| 多 pass 共享与依赖 | 实际 compute、shadow、HDR surface、resolve 与可选 BVH refit 通过 Render Graph 执行；读写和环路验证 | 资源用途/格式与分配管理扩展；临时内存复用 |
| 材质与直接光 | 有界表达式 IR（128 节点、深度 32、16 个 vec4 参数、4 个纹理槽）；PBR/Unlit、Opaque/Masked、双面，sRGB/线性纹理与 mip；基础法线贴图、可负 XY scale 与 glTF 线性 `normalTexture` 槽 3，覆盖程序不依赖法线；映射法线夹在可见几何半球，太阳几何半球约束与着色法线 BRDF 分离；GPU 核对解析 PBR，scale 热更新不增 pipeline、深度/阴影不变 | 扩大运动向量、复杂资产和材质组合验收；GI/反射覆盖在阶段 2 验证，高级透射/层叠仍按路线安排在后续阶段 |
| 物理适配与查询 | Rapier 固定步刚体/碰撞、插值、增删、瞬移；射线/球形范围查询；浏览器查询与渲染联测 | 物理操作的通用接口覆盖与规模性能 |
| 资源与性能诊断 | GPU 分配账本、异步 pass 时间戳、CPU 提交统计；默认 128 个材质变体的未引用 LRU 淘汰和 256 MiB 全 mip 纹理预算；几何页池已接入主渲染器，分配/上传/暂存预算及多池共用提交帧保护；新增按组工作集、祖先保留和 CPU 传输窗口；显式深度诊断；本地 Intel/NVIDIA 短程硬件计时；NVIDIA 基础压力场景 30 分钟账本无增长、释放归零 | 全局资源调度、有界阴影/查询代理、完整帧含呈现计时、GI/LOD 等联合长测；换材质时预算须容纳新旧图像重叠 |
| 可重复验收 | 固定种子/轨迹/压力事件的生产基准入口，帧数或真实时长运行，CSV/JSON/截图及构建指纹；本轮 365 项 CPU（47 文件）及 65 项原生 NVIDIA 浏览器回归通过，新增共享 Worker 解码、取消/重建、初始化期间设备丢失和生产 Worker 入口验证；此前小池往返、整组预算和 GPU 需求验证继续通过；此前 Intel 核显三项流送定向测试、基础压力场景各三次 1800 帧及一次真实 30 分钟运行，运动向量附件数值验证 | 更多厂商/浏览器、复杂法线资产和目标规模的性能档；此前 30 分钟运行的 rAF p95 超出初始帧间隔预算；本轮远程 CI 按新提交验证 |

本次画质回归补充：聚焦阴影的独立斜边误差由 0.19606 降至 0.0003715；单帧 FXAA 保持平坦像素/HDR 不变并减轻蓝白斜边。旧地面六点远场参考的平均误差由 7.97% 降至 5.59%，但使用 0.03 m 起点偏移，不能代表接触区。独立 4096 路径/12 段、0.0001 m 起点偏移的近场参考中，可选实验性 Final Gather 在五个实际像素上的 RGB 平均相对误差从 52.18% 降至 11.29%，最差仍为 30.77%，未达到平均 10% / 单项 20% 目标；1 cm 位置像素命中白块，不参与地面 HDR 对比。此前固定 1080p NVIDIA 小房间三次 GPU pass 总和 p95 为 8.389/9.378/8.982 ms，rAF p95 为 20.8/20.9/20.9 ms，应用 GPU 账本为 123,655,372 字节且释放归零。新相机移动回归中，144 个固定物理点连续帧 RGB 变化 RMS 由逐帧强制清历史的 0.02483 降至重投影的 0.002504，均值各通道差异小于 0.6%；这只验证该相机轨迹的时间稳定性。新轨道镜头 1080p NVIDIA 三次 GPU pass 总和 p95 为 4.656/4.460/4.460 ms，rAF p95 为 20.9/7/7 ms，应用账本 123,852,044 字节稳定且释放归零；不同构建与镜头条件，不能据此推断比固定镜头加速，最差 rAF 仍未通过 60 FPS 门槛。蓝墙此前有限一次反弹参考仍有 23.41% 偏差；更广 GI 画质、性能及完整场景验收继续进行，详见 implementation.md。

## 阶段 2

此前 CPU 流送基础的日志为 `artifacts/geometry-stream-check-release-20261016.log` 和 `artifacts/geometry-stream-browser-release-native-20261016.log`，完整浏览器回归 57 项通过，远程 [CI run 38914399969](https://github.com/fomalhautb/real-engine/actions/runs/38914399969) 也已成功。用例经真实 HTTP gzip/CORS 加载页面，损坏页导致整组祖先保留，显式重试后恢复 288 个 LOD0 三角形并释放 CPU 缓存。

此前物理页池版本的 `artifacts/gpu-page-pool-check-release-20261016.log` 记录 314 项 CPU 测试及构建通过；`artifacts/gpu-page-pool-browser-release-native-20261016.log` 记录实际 `nvidia / ampere` 完整浏览器回归 58 项通过。16 个资产页只分配 6 个槽位、共 484,448 字节，每个上传帧至多 8 KiB；部分页不发布、在途帧阻止淘汰、完成后复用槽位。根页、细节加载和淘汰回退三种实际 GPU 图像分别为 1,068 / 1,452 / 1,198 个三角形，与同一切分的独立完整驻留参考逐像素一致，GPU 验证无错误且释放归零。该同名目录下 `geometry-page-pool-physica-ff17c-only-after-submitted-frames/geometry-page-pool-cuts.png` 是真实截图。诊断使用 CPU 选簇、逐簇 draw 和内存页源，额外持有完整参考几何；当时未接入主 Renderer；当前接入范围见下文，尚未证明整个应用超显存游走。

本轮 `artifacts/gpu-stream-renderer-check-release-20261016.log` 记录 331 项 CPU 测试（44 文件）、类型检查及构建通过；`artifacts/gpu-stream-renderer-browser-release-native-20261016.log` 记录实际 `nvidia / ampere` 完整浏览器回归 61 项通过。新增 GPU 整组切分/缺页请求/物理索引间接绘制与独立参考一致；主渲染器的两个页池、三个实例经自动加载从 1,908 增至 15,360 三角形，五种材质/运动/GI 状态的 768 个 HDR 抽样最大误差均为零，有限查询、运动向量、深度/阴影与参考一致，释放后账本为零。主渲染器截图在同名结果目录下 `geometry-stream-renderer-m-4218a-adow-query-and-GI-semantics/streamed-main-renderer.png`。Intel 核显另通过三项流送定向测试。这些证据尚不覆盖超驻留池的大场景、动态简化或目标性能，完整范围见 implementation.md。

本轮组级调度通过 346 项 CPU（45 文件）与 62 项原生 NVIDIA 浏览器回归，日志为 `artifacts/stream-working-set-check-release-20261019.log` 和 `artifacts/stream-working-set-browser-release-native-20261019.log`。16 页资产在 8 槽、644,384 字节的物理池内，A→B→A→B→A 完成四次细节页更替，根页和祖先路径保留；160 帧实际切分与独立参考一致，五个稳定状态后各 12 帧无重复上传/请求。CPU 解码驻留加预留峰值 205,448 字节，低于 205,848 字节预算。独立数学参考还核对 GPU 组需求的仿射实例最大值、实例顺序、驻留无关、冻结视点和离屏剔除。截图见同名浏览器目录的 `geometry-stream-budget-sma-ecf19-peated-uploads-across-views/stream-working-set.png`。这仍非全应用超显存、大资产或性能验证。

本轮 Worker 页解码通过 365 项 CPU（47 文件）与 65 项完整原生 NVIDIA 浏览器回归，日志为 `artifacts/geometry-worker-check-release-20261019.log` 和 `artifacts/geometry-worker-browser-release-native-20261019.log`。两个资产批次共用两个 Worker 完成 42 次解码，在途载荷峰值 321,344 字节；取消/释放后活动项和在途字节归零。新增真实 Worker 转移与逐字节核对、取消重建、显式设备丢失以及根页初始化中止；生产 Worker 构建入口另通过同一生命周期用例。目录身份校验、源属性核对与上传打包仍在主线程；此结果不证明通用资产导入已迁到 Worker、全应用内存有界或性能达标，详见 implementation.md。

下列工作包仍未完整交付，不因局部功能的测试通过而视作已达标。跨帧漫反射反馈、阴影修正和实体墙演示场景已在提交 `552a8dc` 的[远程回归 run 38526339749](https://github.com/fomalhautb/real-engine/actions/runs/38526339749) 验证（215 项 CPU、27 项浏览器测试）；保存同镜头单次/多次反弹及 GI 开关截图。明显的直接阴影条纹和墙根长亮带已修正，粗粒度 GI 的块状瑕疵和亮度连续性仍待处理。

| 工作包 | 必须交付的实现与证据 |
| --- | --- |
| 离线几何构建 | 已有静态 Opaque 簇 DAG、边界锁定的真实简化、累计近似误差、CPU 参考切分、压缩页面与 OBJ 构建入口；仍需大型真实资产、更多拓扑/属性和形变/Masked 的误差与边界验收 |
| GPU 可见性 | 已接入静态 Opaque 的驻留与分页 GPU LOD、视锥剔除、索引压紧与间接绘制；分页选择按真实驻留状态整组回退并在 GPU 合并缺页请求，多个仿射实例/冻结视点/视锥外/零实例回归通过；提供实际簇稳定分色、LOD 层级、固定轨迹/暂停逐帧演示及分页加载/淘汰诊断；仍需遮挡剔除、层级遍历优化、动态形变和更大场景验证 |
| 几何流送 | 已有离线独立压缩页面、紧凑目录及哈希；CPU 缓存完成根页固定、请求合并、并发/字节预算、LRU、损坏拒绝与显式重试；物理 GPU 页池完成分配、分帧上传、整页发布及多池共享提交帧保护；GPU 需求驱动主 Renderer 静态 Opaque 表面按整组预算加载并保留祖先路径，CPU 传输窗口留住待上传数据，取消过期下载/未发布上传；复用材质/法线贴图/当前前帧变换，阴影、查询与 Masked 保留完整源网格；新增共享 Worker 哈希/解码、转移缓冲、下载前预留、取消/故障恢复和设备丢失清理；仍需有界查询/阴影代理、全局预算与动态简化；独特资产为池 4–8 倍仍可游走且不破洞尚未验收 |
| 世界空间查询 | 可选 GPU BVH refit/最近命中基础，读取共享形变与材质覆盖，含实体/generation/历史/材质版本；此前法线贴图回归在不扩大 160 字节 `WorldHit` 的前提下返回贴图着色法线与真实三角面 `geometricNormal`，有限平面回归验证查询仍读取完整源几何；同一 encoder 用不可变对齐槽位支持不同 count/LOD 参数（默认 8 组、硬上限 32，超额显式报错），仍要求相同帧/材质/批次数/拓扑快照；仍需实例加速/TLAS、大场景吞吐与更新成本，以及 BVH/距离场实验选型 |
| 动态漫反射 GI | 已接入 GPU 固定网格、SH9 入射辐亮度、8×8 八面体方向深度矩、可见性插值、世界查询太阳遮挡及可选跨帧反馈。面积查询处理活跃前缀；此前演示第三轮 6,912 条，世界 GI 总查询 62,208 条。可选实验性屏幕 Final Gather 使用实际像素命中、首段余弦采样与发光面积样本、独立原始历史和过滤输出；此前法线贴图回归接入几何半球截断、几何法线偏移及历史/过滤平面，60° 均匀天空的四个原始锚点相对 `E = π × L × 0.75`、五个 HDR 点相对 `.96 × baseColor × L × 0.75` 的误差各小于 5%；普通相机移动对前帧原始锚点重投影并将有效历史限于 8 帧，相机切换、动态批次或光照/材质等依赖改变仍整体清历史。世界 SH 仍按着色法线做完整余弦卷积，未精确截掉几何背侧，不能宣称完整 GI 正确；近场质量目标、复杂漏光、离屏稳态、增量更新及大场景预算/性能验收仍待完成 |
| 反射 | 粗糙反射和受控光滑反射；离屏角色/刚体可见，公开质量档与近似 |
| 体积雾与体积光照 | 已加入线性 HDR 合成路径和 CPU 均匀介质解析参考，尚无 GPU 体积雾；阶段 2 中期交付高度雾/局部体积、太阳/点光/聚光灯与几何阴影、低频 GI、时序重建和已支持反射中的介质路径；后期通过 F01–F05 与大型场景验收 |
| 缓存更新 | 当前 GI 每帧读取最新查询与材质并全量刷新探针；可选双场跨帧反馈保留上一完成帧的 SH，依赖变化时重置。仍需分级缓存、变更驱动与优先级更新、资源页面失效及最大陈旧度控制 |
| 联合验收 | 开关灯、开门、移动物体的 T90/历史恢复；漏光、稳态误差、时间序列和性能预算 |
| 大规模完整场景与视觉交付 | 阶段 2 后期交付高质量室内/室外联合场景；候选 Sponza＋扩展、Bistro、Jungle Ruins；固定镜头/参考图、连续游走录屏、动态角色与物体、帧时间/内存及用户视觉验收，详见 benchmarks.md。每次新增可见成果在对话中展示实际截图或视频，并注明实现范围 |

此前 glTF 切线生成证据在 `artifacts/mikk-tangents-check-20261015.log`、`artifacts/mikk-tangents-browser-native-20261015.log`。CPU 回归覆盖按需 WASM 初始化及失败路径、工作量预算、镜像 UV 接缝拆分、基础与全部 Morph 目标切线生成和各顶点流重映射。原生 GPU 用例将实际 glTF 的 4 个输入顶点拆为 6 个输出顶点、绘制 2 个三角形，联测生成的 Morph 切线、仿射蒙皮、法线贴图解析 PBR/HDR、世界查询的映射/几何法线；位置/法线/切线验证误差小于 `3e-4`，边界越界为零、GPU 错误为空、释放后应用账本为零。中性与映射截图为 `artifacts/mikk-tangents-browser-native-20261015/gltf-tangents-glTF-generat-ea424-skin-raster-and-world-query/gltf-tangents-neutral.png`、同目录的 `gltf-tangents-mapped.png`。这证明所测资产的功能路径，不代表参考 MikkTSpace 的逐位一致性或复杂资产性能。

此前 release 法线贴图证据在 `artifacts/normal-map-check-release-20261015.log`、`artifacts/normal-map-browser-release-native-20261015.log`；GPU 用例覆盖解析 PBR、源/映射法线、背面与镜像、共享形变和 UV 回退、scale 热更新、几何半球、128 次×8 射线的均匀天空 Final Gather、微小三角形有限性，以及 512→102 三角形/5 簇的实际 Opaque LOD 联合路径。替换 `mesh.tangents` 引用明确报 `WORLD_QUERY_TOPOLOGY_CHANGED`，恢复原引用后查询正常；释放后应用账本为零、错误为空。原生截图位于 `artifacts/normal-map-browser-release-native-20261015/material-normal-authored-n-e8268-eserve-geometric-visibility/`，其中有 `normal-map-neutral.png` 和 `normal-map-tilted.png`。这只证明所测有限平面，不覆盖任意复杂法线贴图资产。

此前三轮硬件镜头基准在 `artifacts/benchmarks/local-normal-layout-release-camera-20261015`：Edge 155 的实际非 fallback `nvidia / ampere`，1920×1080、相同 orbit/Final Gather，300 帧预热＋1800 帧测量，每轮 60 个稀疏 GPU 样本。GPU pass 总和 p95 为 4.5907968/4.5252608/4.718592 ms，Final Gather 为 2.1659648/2.031616/2.031616 ms，世界 GI 为 1.96608/1.96608/2.031616 ms；rAF p95 为 7.0/7.0/13.9 ms、CPU 提交 p95 为 0.9/0.8/0.5 ms。应用账本稳定为 123,856,236 字节，释放归零且错误为空。此 GI 房间未启用 `normalTexture`，只验证 64 字节共享布局和原有 GI 的基础回归；不能证明新 glTF 切线生成成本、复杂法线贴图开销、大场景性能或稳定 60 FPS，也不能跨旧构建用分位数推断加速。

完成判定需要逐项实现和对应范围的实际验证。软件适配器测试仅证明测试覆盖范围内的功能正确性，不能代替性能、画质或长期稳定性门槛。

2026-10-14 的本地基础基准使用 Edge、128 棵树、两个官方动画资产、物理事件与动态分辨率。默认 `intel / gen-12lp` 的 GPU pass 总和 p95 为 137–147 ms；强制高性能选择 `nvidia / ampere` 后为 0.918–1.376 ms。旧六方向距离矩版的固定 1080p GI 小房间三次总和 p95 为 2.300/2.104/2.425 ms、GI pass 为 0.983/0.996/0.993 ms；新 8×8 方向可见性与置信度版在同样房间三次总和为 2.297/2.032/2.100 ms、GI pass 为 1.124/1.114/1.052 ms，账本稳定为 39,805,004 字节并释放归零。固定 1080p 几何演示三次总和 p95 为 0.872/0.852/0.809 ms，LOD pass 为 0.197 ms；预热后账本稳定、释放归零。演示基准都核对了 Renderer 实际 NVIDIA adapter。几何测试的 165 个实际选中簇均为 LOD0，三个实例实际绘制的 15,360 个雕塑三角形等于源量，尚未在此镜头证明减面收益。证据在 `artifacts/benchmarks/local-foundation-20261014`、`local-foundation-nvidia-20261014`、`local-gi-nvidia-20261014-v3`、`local-gi-directional-nvidia-20261014` 和 `local-geometry-nvidia-20261014-v3`；各 manifest 记录自身源码/构建指纹，新旧 GI 使用不同构建。这些是有限场景的稀疏 GPU 时间戳，不是目标内容规模下的整帧/呈现耗时或阶段出口。

另一次干净源码 `eda4d5c` 的 NVIDIA 基础压力场景运行了 1,800,000.2 ms、167,349 帧及 92 个完整事件循环，应用 GPU 账本和分段中位数均无增长，释放为零，错误为空，25,016 个顶点验证通过；此范围的 **30 分钟资源/功能稳定性通过**。但 rAF p95 为 34.8 ms、p99 62.5 ms，10,401 帧超出 33.3 ms 预算，性能门槛**未通过**。动态分辨率、稀疏 GPU 时间戳与少量同期短 CPU 测试限制了性能归因；并未对 GI、LOD、反射、体积雾或完整目标场景进行长测。证据目录为 `artifacts/benchmarks/local-stability-nvidia-20261014`，其源码/构建指纹独立于上述短程基准。
