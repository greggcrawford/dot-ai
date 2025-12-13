import { ConsoleLogger, Logger } from '../error-handling';

export class RateLimitError extends Error {
  public readonly retryAfterMs?: number;

  constructor(message: string, retryAfterMs?: number) {
    super(message);
    this.name = 'RateLimitError';
    this.retryAfterMs = retryAfterMs;
  }
}

export interface AdaptiveRateLimiterOptions {
  /**
   * Minimum requests per second the limiter will allow after backoff.
   */
  minRps?: number;
  /**
   * Maximum requests per second once the limiter fully recovers.
   */
  maxRps?: number;
  /**
   * Initial requests per second when the limiter starts.
   */
  initialRps?: number;
  /**
   * Maximum burst size (token bucket capacity).
   */
  burstCapacity?: number;
  /**
   * Factor applied to current RPS when a rate limit error occurs.
   */
  backoffMultiplier?: number;
  /**
   * Increment applied to current RPS after sustained success.
   */
  recoveryStep?: number;
  /**
   * Number of successful requests before applying recovery step.
   */
  successesForRecovery?: number;
  /**
   * Minimum backoff wait (ms) when Retry-After header is absent.
   */
  minBackoffMs?: number;
  /**
   * Maximum backoff wait (ms) when Retry-After header is absent.
   */
  maxBackoffMs?: number;
  /**
   * Logger instance; defaults to ConsoleLogger with component AdaptiveRateLimiter.
   */
  logger?: Logger;
}

interface QueueEntry<T> {
  task: () => Promise<T>;
  resolve: (value: T | PromiseLike<T>) => void;
  reject: (reason?: any) => void;
}

const DEFAULT_OPTIONS: Required<Omit<AdaptiveRateLimiterOptions, 'logger'>> = {
  minRps: 0.5,
  maxRps: 5,
  initialRps: 1,
  burstCapacity: 5,
  backoffMultiplier: 0.5,
  recoveryStep: 0.25,
  successesForRecovery: 30,
  minBackoffMs: 1_000,
  maxBackoffMs: 60_000
};

export class AdaptiveRateLimiter {
  private readonly options: Required<Omit<AdaptiveRateLimiterOptions, 'logger'>>;
  private readonly logger: Logger;

  private currentRps: number;
  private tokens: number;
  private lastRefill: number;
  private queue: QueueEntry<any>[] = [];

  private pausedUntil: number | null = null;
  private processingTimer: ReturnType<typeof setTimeout> | null = null;
  private currentBackoffMs: number;
  private successStreak = 0;

  constructor(options: AdaptiveRateLimiterOptions = {}) {
    this.options = { ...DEFAULT_OPTIONS, ...options };
    this.logger = options.logger || new ConsoleLogger('AdaptiveRateLimiter');
    this.currentRps = Math.min(
      Math.max(this.options.initialRps, this.options.minRps),
      this.options.maxRps
    );
    this.tokens = this.options.burstCapacity;
    this.lastRefill = Date.now();
    this.currentBackoffMs = this.options.minBackoffMs;
  }

  /**
   * Schedule a task to run under rate limiting.
   */
  async schedule<T>(task: () => Promise<T>): Promise<T> {
    return await new Promise<T>((resolve, reject) => {
      const entry: QueueEntry<T> = { task, resolve, reject };
      this.enqueue(entry);
    });
  }

  private enqueue(entry: QueueEntry<any>, delayMs: number = 0): void {
    if (delayMs > 0) {
      setTimeout(() => {
        this.queue.push(entry);
        this.processQueue();
      }, delayMs);
    } else {
      this.queue.push(entry);
      this.processQueue();
    }
  }

  private refillTokens(): void {
    const now = Date.now();
    const elapsedSeconds = (now - this.lastRefill) / 1000;
    if (elapsedSeconds <= 0) {
      return;
    }
    this.lastRefill = now;
    this.tokens = Math.min(
      this.options.burstCapacity,
      this.tokens + elapsedSeconds * this.currentRps
    );
  }

  private processQueue(): void {
    if (this.processingTimer) {
      return;
    }

    const step = () => {
      this.processingTimer = null;

      if (this.pausedUntil && Date.now() < this.pausedUntil) {
        const waitMs = this.pausedUntil - Date.now();
        this.scheduleNextProcessing(waitMs);
        return;
      } else if (this.pausedUntil && Date.now() >= this.pausedUntil) {
        this.pausedUntil = null;
      }

      this.refillTokens();

      while (this.queue.length > 0 && this.tokens >= 1) {
        const entry = this.queue.shift();
        if (!entry) break;
        this.tokens -= 1;
        this.executeEntry(entry);
      }

      if (this.queue.length > 0) {
        const waitMs = this.timeUntilNextToken();
        if (waitMs !== null) {
          this.scheduleNextProcessing(waitMs);
        }
      }
    };

    this.processingTimer = setTimeout(step, 0);
  }

  private timeUntilNextToken(): number | null {
    if (this.tokens >= 1) {
      return 0;
    }
    const deficit = 1 - this.tokens;
    if (this.currentRps === 0) {
      return 1000;
    }
    return Math.max(0, (deficit / this.currentRps) * 1000);
  }

  private scheduleNextProcessing(delayMs: number): void {
    if (this.processingTimer) {
      clearTimeout(this.processingTimer);
    }
    this.processingTimer = setTimeout(() => {
      this.processingTimer = null;
      this.processQueue();
    }, delayMs);
  }

  private executeEntry(entry: QueueEntry<any>): void {
    entry
      .task()
      .then(result => {
        this.onSuccess();
        entry.resolve(result);
      })
      .catch(error => {
        if (error instanceof RateLimitError) {
          this.onRateLimit(error.retryAfterMs);
          const retryDelay =
            error.retryAfterMs ??
            Math.min(
              this.currentBackoffMs,
              this.options.maxBackoffMs
            );
          this.enqueue(entry, retryDelay);
        } else {
          entry.reject(error);
        }
      })
      .finally(() => {
        this.processQueue();
      });
  }

  private onSuccess(): void {
    this.successStreak += 1;
    if (
      this.successStreak >= this.options.successesForRecovery &&
      this.currentRps < this.options.maxRps
    ) {
      this.currentRps = Math.min(
        this.options.maxRps,
        this.currentRps + this.options.recoveryStep
      );
      this.successStreak = 0;
      this.currentBackoffMs = this.options.minBackoffMs;
      this.logger.info('Adaptive rate limiter increased throughput', {
        currentRps: this.currentRps
      });
    }
  }

  private onRateLimit(retryAfterMs?: number): void {
    this.successStreak = 0;
    this.currentRps = Math.max(
      this.options.minRps,
      this.currentRps * this.options.backoffMultiplier
    );
    if (retryAfterMs && retryAfterMs > 0) {
      this.pausedUntil = Date.now() + retryAfterMs;
    }
    this.currentBackoffMs = Math.min(
      Math.max(this.currentBackoffMs * 2, this.options.minBackoffMs),
      this.options.maxBackoffMs
    );
    this.logger.warn('Rate limit encountered, reducing throughput', {
      currentRps: this.currentRps,
      retryAfterMs: retryAfterMs ?? this.currentBackoffMs
    });
  }
}
