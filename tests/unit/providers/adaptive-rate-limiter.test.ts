import { describe, it, expect, vi, afterEach } from 'vitest';

import {
  AdaptiveRateLimiter,
  RateLimitError,
} from '../../../src/core/providers/adaptive-rate-limiter';

describe('AdaptiveRateLimiter', () => {
  afterEach(() => {
    vi.clearAllTimers();
    vi.useRealTimers();
  });

  it('executes queued tasks in order under normal conditions', async () => {
    const limiter = new AdaptiveRateLimiter({
      initialRps: 10,
      burstCapacity: 10,
      minRps: 1,
      maxRps: 10,
    });

    const executionOrder: number[] = [];
    const results = await Promise.all(
      [1, 2, 3].map(value =>
        limiter.schedule(async () => {
          executionOrder.push(value);
          return value * 2;
        }),
      ),
    );

    expect(results).toEqual([2, 4, 6]);
    expect(executionOrder).toEqual([1, 2, 3]);
  });

  it('backs off and retries when rate limits are encountered', async () => {
    const limiter = new AdaptiveRateLimiter({
      initialRps: 1,
      minRps: 0.5,
      maxRps: 2,
      burstCapacity: 1,
      minBackoffMs: 5,
      maxBackoffMs: 20,
      successesForRecovery: 2,
    });

    let attempts = 0;
    const task = vi.fn().mockImplementation(async () => {
      attempts += 1;
      if (attempts === 1) {
        throw new RateLimitError('limit', 50);
      }
      return 'ok';
    });

    await expect(limiter.schedule(task)).resolves.toBe('ok');
    expect(task).toHaveBeenCalledTimes(2);
  });
});
