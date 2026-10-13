# 调研与技术选择依据

调研基线：2026-10-13；项目决策更新：2026-10-14。Unreal 对照采用当前官方 UE 5.8 文档，正式验收还必须固定补丁版本与配置。本文区分官方已描述能力、工程推断和待测决策，不包含本项目性能实测。

## 目标重新定义

WebGPU 是图形与计算 API，Unreal 是包含资源构建、渲染、物理、内容工具和运行时的引擎，两者不能直接按材质数量或画质百分比比较。本项目实际要建设的是 WebGPU 上的引擎系统，借鉴 Nanite 的虚拟几何、Lumen 的世界空间动态光照及 Unreal 的材质组织能力。

项目面向角色与大量动态物体，不能沿用“主要静态室内场景，角色以后再说”的范围。几何、形变、材质和光照应共同立项；代码优先免去编辑器产品线，但不会使资源流送、编译优化和跨系统一致性自动成立。80% 采用 [验收协议](benchmarks.md) 中的加权用例矩阵，而不是对 Unreal 整体能力的无定义承诺。

## Nanite：价值、优先级与边界

Nanite 的核心组合是分层几何簇、屏幕误差驱动的细节选择、GPU 剔除、微三角形渲染、压缩和细粒度流送。它不是只有自动减面；大量独特资产超过驻留内存后仍可使用，才体现虚拟几何价值。材质复杂度、输出分辨率和实例规模仍有实际成本。[官方概览](https://dev.epicgames.com/documentation/unreal-engine/nanite-virtualized-geometry-in-unreal-engine)、[原始技术分享](https://advances.realtimerendering.com/s2021/index.html)

| 能力 | 本项目优先级 | 关键验证 |
| --- | --- | --- |
| 簇层级、误差、边界一致 | 核心必需 | 近远切换无裂缝，轮廓误差可测 |
| GPU 实例/簇剔除与间接执行 | 核心必需 | 快速转头正确恢复，成本随可见工作变化 |
| 分页、压缩、淘汰与缺页回退 | 核心必需 | 独特数据超预算后持续运行 |
| 刚体、共享蒙皮/Morph、材质关联 | 通用引擎核心必需 | 动态状态在阴影和查询中一致 |
| 微三角形软件光栅 | 重要优化，先测 | 对比硬件基线的收益与适用区间 |
| 动态细分、复杂拓扑变化 | 后续高级能力 | 接缝、边界、代理及历史更新 |

不能把早期“Nanite 不支持骨骼”当当前结论；5.8 官方概览已有骨骼 API、动画 LOD 和动画银行实例化。当前材质支持仍有明确边界，例如 Opaque/Masked、WPO 位移界限，以及透明 mesh decal 等特殊路径。因此项目可以追求更统一的合约，但不能宣称 Unreal 自己已支持任意组合。

动态 Nanite tessellation 在内容文档仍标为实验性：标量法线位移、与 PDO 不兼容，属性不连续处仍有裂缝问题。材质位移需要几何算法配合，不能当普通参数开关。[Nanite 内容工作流](https://dev.epicgames.com/documentation/unreal-engine/working-with-naniteenabled-content)

资产构建优先调查并复用 [meshoptimizer](https://github.com/zeux/meshoptimizer) 的簇、简化与压缩能力；[vk_lod_clusters](https://github.com/nvpro-samples/vk_lod_clusters) 可参考层级与流送组织；[nanite-webgpu](https://github.com/Scthe/nanite-webgpu) 可参考浏览器实现和作者公开的限制。它们不是已验证满足本项目动态材质与光照需求的整体解决方案。

## Lumen：必须保住离屏世界

Lumen 的价值是动态漫反射反弹、天空遮蔽、自发光贡献和反射；离开画面的物体仍然能够参与照明和遮挡。应区分三个系统：几何追踪回答命中位置，Surface Cache 提供表面材质/光照，探针及辐射缓存降低采样开销。屏幕空间追踪只能补充细节，不能替代完整离屏贡献。[Lumen 功能](https://dev.epicgames.com/documentation/en-us/unreal-engine/lumen-global-illumination-and-reflections-in-unreal-engine)

本项目先比较 Compute BVH 与距离场：前者便于检验命中、薄结构和动态代理，后者适合空间遮挡与部分 GI 查询。选择依据必须包含形变更新、查询吞吐、存储和漏光误差。随后建立遮挡感知探针、动态 GI、粗糙反射及受控光滑反射，可参考 [DDGI 原论文](https://research.nvidia.com/publication/2019-05_dynamic-diffuse-global-illumination-ray-traced-irradiance-fields)。

UE 软件距离场路径不支持 WPO 形变；硬件追踪支持更广的动态几何。5.8 骨骼 Surface Cache 默认启用，但使用蒙皮前包围盒的六张 card，剧烈形变时覆盖受限。视角依赖材质也需为缓存提供合适评估。这些事实说明“接收 GI”“进入缓存”“离屏射线能命中”是不同能力，必须分别验收。[Lumen 技术细节](https://dev.epicgames.com/documentation/en-us/unreal-engine/lumen-technical-details-in-unreal-engine)

Lumen Medium/Lite 使用更便宜的探针插值；平滑表面采用 SSR，不能把该档看作完整离屏反射的同等替代。本项目可借鉴其质量分级，但动态世界查询仍是目标要求。[Lumen 性能指南](https://dev.epicgames.com/documentation/en-us/unreal-engine/lumen-performance-guide-for-unreal-engine)

Virtual Shadow Maps、虚拟纹理、时域超分和世界流送是相邻独立系统，不应全部计入“实现 Lumen”。直接阴影先建立正确性基线，再依据测量升级。[Virtual Shadow Maps](https://dev.epicgames.com/documentation/en-us/unreal-engine/virtual-shadow-maps-in-unreal-engine)

## 材质差距要分三层看

**视觉公式层**：PBR、法线、Clear Coat、各向异性、布料 Fuzz、薄膜等可用着色器实现，WebGPU 没有天然禁止这些外观。皮肤、发丝、眼睛和透明则往往需要专用空间算法、几何约定和额外 pass；仅完成 BRDF 不能代表对应完整系统。UE 有 Unlit、Default Lit、SSS/Skin、Clear Coat、Foliage、Hair、Cloth、Eye、Water、Thin Translucent 等模型。[Shading Models](https://dev.epicgames.com/documentation/unreal-engine/shading-models-in-unreal-engine?lang=en-US)、[皮肤散射](https://dev.epicgames.com/documentation/en-us/unreal-engine/subsurface-profile-shading-model-in-unreal-engine)

**规模与性能层**：材质实例、变体数量、纹理驻留、过度绘制、多层光学、首次编译、形变及 GI 捕获共同决定成本。参数实例可避免编译，但静态开关的不同组合仍产生变体；必须进行编译期求值、变体管理和异步预热。[Material Instances](https://dev.epicgames.com/documentation/unreal-engine/instanced-materials-in-unreal-engine)、[PSO Precaching](https://dev.epicgames.com/documentation/unreal-engine/pso-precaching-for-unreal-engine)

**制作系统层**：UE 的图、函数、实例、分析和资产管线是重要工程积累。本项目不复制编辑器，但需要代码 DSL/IR、诊断、依赖分析、质量回退和成本观测。没有 UI 不等于没有编译器。[Material Functions](https://dev.epicgames.com/documentation/en-us/unreal-engine/unreal-engine-material-functions-overview)

当前 Substrate 新项目默认启用。普通模板的 Blendable GBuffer 偏向固定内存和性能；Adaptive GBuffer 支持更复杂材质，并按 closure 数和存储预算简化。层叠会增加计算与带宽，特定 Hair/Eye/Water BSDF 也不能任意参与 Slab 组合。建议固定模型与类型化 IR 起步，再加入有界 layering；不从任意层运行时解释器开始。[Substrate 总览](https://dev.epicgames.com/documentation/unreal-engine/overview-of-substrate-materials-in-unreal-engine)

发丝不仅是材质：Groom 还包含模拟、插值、体素化、可见性、阴影和合成；发片/网格仍是官方质量缩放路径。布料材质与布料模拟同样应区分。[Groom 管线与成本](https://dev.epicgames.com/documentation/unreal-engine/groom-scalability-and-performance-with-unreal-engine)

纹理虚拟化与 decal 也需要资源和 pass。SVT 按需加载离线页面，RVT 运行时生成缓存；它们节省驻留空间但不免除页表查询与采样开销。[虚拟纹理](https://dev.epicgames.com/documentation/unreal-engine/virtual-texturing-in-unreal-engine)、[Decal](https://dev.epicgames.com/documentation/en-us/unreal-engine/decal-materials-in-unreal-engine)

## WebGPU 的实际约束

现有 Compute、storage buffer、纹理和间接绘制足以支持自有几何、形变与材质系统；WGSL 可以承载其数学和控制逻辑。但浏览器标准不能直接假定具有原生硬件光追、Mesh/Task Shader 或通用 bindless。可选能力和正在推进的实验能力必须运行时探测，不能用原生 wgpu 或实验浏览器的存在证明所有用户可用。[GPUWeb 规范](https://gpuweb.github.io/gpuweb/)、[WGSL 规范](https://gpuweb.github.io/gpuweb/wgsl/)

工程推断：主要风险在世界空间软件查询、动态结构更新、资源绑定/调度和浏览器内存/加载，而非“能否写一个金属着色器”。自定义任意 WGSL 与可分析、可融合、可复用之间存在真实冲突，应以明确扩展合约解决，详见 [architecture.md](architecture.md)。

## Three.js、Babylon.js 与自有核心

Three.js 已提供 WebGPURenderer、计算和间接能力，不能简单说它无法做高级渲染；Babylon.js 的 FrameGraph 支持自定义任务，Babylon-Lite 也以 WebGPU 和更精简结构为方向。它们适合作为能力参考、原型对照和生态适配候选。

本项目选择自有核心，是为了让场景变更、统一形变、虚拟几何、追踪和材质共享明确合约，控制调度与资源生命周期；这不是“自研天然更快”的结论。自有核心增加正确性、工具、兼容与维护负担，应保留小型对照场景测量，而不是凭抽象层数判断性能。

依据：[Three WebGPURenderer](https://threejs.org/manual/pages/webgpurenderer)、[Three Renderer](https://threejs.org/docs/pages/Renderer.html)、[Three BufferGeometry](https://threejs.org/docs/pages/BufferGeometry.html)、[Babylon FrameGraph 自定义任务](https://doc.babylonjs.com/features/featuresDeepDive/frameGraph/frameGraphClassFramework/frameGraphWritingTasks/)、[Babylon-Lite 设计理念](https://github.com/BabylonJS/Babylon-Lite/blob/master/philosophy.md)。以上能力说明不构成本项目的性能排名。

## 物理引擎选择

阶段 1 采用 [Rapier JavaScript/WASM](https://rapier.rs/docs/user_guides/javascript/getting_started_js/) 的适配器，先把固定步、变换同步、查询和基本碰撞做成可验证切片。[JoltPhysics.js](https://github.com/jrouwe/JoltPhysics.js/) 作为不同刚体规模的候选；若未来改用 Babylon 主体，[Havok 插件](https://doc.babylonjs.com/features/featuresDeepDive/physics/havokPlugin/) 是现成集成路径。

本轮没有同场景物理性能实测，不宣称 Rapier 比 Jolt/Havok 更快。选择应依据包体、启动、更新成本、接触/约束规模、查询、稳定性和平台覆盖重新评估。渲染微三角形与碰撞体分离，不能让相机 LOD 改变碰撞行为；两者通过场景身份、单位和版本保持一致。

## 基准与待验证结论

本次没有发现一个可直接计算“Nanite/Lumen 完成 80%”的官方通用分数。采用公开资产、自建微场景和固定动态轨迹：几何看原始 LOD0，光学看收敛参考，产品能力看冻结的 UE 配置。[Intel Sponza](https://www.intel.com/content/www/us/en/developer/topic-technology/graphics-research/samples.html)、[Blender 示例](https://www.blender.org/download/demo-files/) 可作为资产来源；[FLIP](https://research.nvidia.com/publication/flip) 可辅助图像差异分析，但不能替代动态和局部错误检查。

优先验证四件事：共享形变能否减少跨 pass 重算；动态世界查询能否保留帧预算；独特资产流送能否稳定限内存；材质变化是否正确更新所有消费者。详细场景、目标与统计协议见 [benchmarks.md](benchmarks.md)，阶段交付见 [roadmap.md](roadmap.md)。
