export type EntityId = number;

/** Entity IDs are never reused, so a stale reference cannot identify a new entity. */
export class EntityRegistry {
  private nextId = 1;
  private readonly entities = new Set<EntityId>();

  create(): EntityId {
    if (!Number.isSafeInteger(this.nextId)) {
      throw new RangeError('Entity ID space exhausted');
    }
    const id = this.nextId++;
    this.entities.add(id);
    return id;
  }

  remove(id: EntityId): void {
    this.entities.delete(id);
  }

  has(id: EntityId): boolean {
    return this.entities.has(id);
  }
}

export type StepResult = {
  steps: number;
  /** Fraction of one fixed step still accumulated, for rendering interpolation. */
  alpha: number;
  /** Simulation time discarded by the backlog cap in this advance call. */
  droppedSeconds: number;
};

export class FixedStepper {
  private accumulator = 0;
  private isPaused = false;

  constructor(
    readonly dt = 1 / 60,
    readonly maxSteps = 5,
  ) {
    if (!Number.isFinite(dt) || dt <= 0) {
      throw new RangeError('dt must be finite and positive');
    }
    if (!Number.isSafeInteger(maxSteps) || maxSteps < 1) {
      throw new RangeError('maxSteps must be a positive safe integer');
    }
    if (!Number.isFinite(dt * maxSteps)) {
      throw new RangeError('dt * maxSteps must be finite');
    }
  }

  get paused(): boolean {
    return this.isPaused;
  }

  pause(): void {
    this.isPaused = true;
  }

  resume(): void {
    this.isPaused = false;
  }

  reset(): void {
    this.accumulator = 0;
  }

  advance(elapsed: number, step: (dt: number) => void): StepResult {
    if (!Number.isFinite(elapsed) || elapsed < 0) {
      throw new RangeError('elapsed must be finite and nonnegative');
    }
    if (this.isPaused) {
      return { steps: 0, alpha: this.accumulator / this.dt, droppedSeconds: 0 };
    }

    const total = this.accumulator + elapsed;
    // Bound the amount simulated after a stall. Include old fractional time in
    // this calculation so the reported discarded time is exact for this call.
    const capped = Math.min(total, this.dt * this.maxSteps);
    const droppedSeconds = Math.max(0, total - capped);
    const tolerance = this.dt * 1e-9;
    const steps = Math.min(this.maxSteps, Math.floor((capped + tolerance) / this.dt));
    this.accumulator = Math.max(0, capped - steps * this.dt);
    for (let i = 0; i < steps; i++) step(this.dt);
    return { steps, alpha: this.accumulator / this.dt, droppedSeconds };
  }
}
