# Repository guidance

This is a code-first WebGPU engine. Read docs/architecture.md and docs/roadmap.md before extending the runtime.

- Keep scene IDs, geometry, deformation, materials, lighting, and physics interfaces independent of Three.js and Babylon.js object types.
- Share deformation operators across characters, vegetation, and other content. Do not introduce asset-category-specific rendering pipelines.
- Separate simulation state, current rendered state, and previous rendered state. Make topology changes and history invalidation explicit.
- Physics collision assets must not depend on camera-selected rendering LOD. Avoid per-frame GPU-to-CPU geometry readback.
- Keep WebGPU validation enabled. Dispose owned GPU/WASM resources and handle initialization failures and device loss explicitly.
- Implement algorithms with bounded workloads and typed contracts. Do not add an editor, runtime AI inference, or gameplay frameworks to the rendering core.
- Update docs/implementation.md when behavior or milestone status changes. Never report planned GI, virtual geometry, or benchmark targets as implemented results.
- Validate with npm run check and, when touching GPU paths, npm run test:browser. Software-adapter tests prove functional behavior, not hardware performance.
- Use the package lock; no secrets or generated build/test output in commits.
