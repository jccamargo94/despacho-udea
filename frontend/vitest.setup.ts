import "@testing-library/jest-dom/vitest";

// jsdom has no layout engine, so Recharts' ResponsiveContainer (which sizes
// itself via ResizeObserver) never measures a non-zero size and renders
// nothing. Stub it to synchronously report a fixed size on observe().
class ResizeObserverStub {
  private cb: ResizeObserverCallback;
  constructor(cb: ResizeObserverCallback) {
    this.cb = cb;
  }
  observe(target: Element) {
    this.cb(
      [{ target, contentRect: { width: 500, height: 320 } } as ResizeObserverEntry],
      this as unknown as ResizeObserver
    );
  }
  unobserve() {}
  disconnect() {}
}
globalThis.ResizeObserver = ResizeObserverStub as unknown as typeof ResizeObserver;
