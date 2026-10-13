# Real Engine

面向代码与 AI 编程的 WebGPU 游戏引擎。核心方向是可组合的虚拟几何、动态全局光照、材质与形变系统，并通过明确的数据接口集成物理。

## 项目计划

- [架构与接口](docs/architecture.md)
- [研发阶段与交付门槛](docs/roadmap.md)
- [基准测试与验收](docs/benchmarks.md)
- [技术调研与一手来源](docs/research.md)

## 当前状态

项目处于第一阶段启动期。计划中的 Nanite 类虚拟几何、Lumen 类世界空间 GI 和反射均尚未实现，文档中的性能数值是待验证目标。

第一批实现以原生 WebGPU 和 TypeScript 为基础，验证程序化角色与树木共用 GPU 蒙皮，以及材质、阴影、运动历史和固定步长物理的连接。具体实现与验证状态将在首个代码版本中记录。
