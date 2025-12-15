import {
  AIProvider,
  AIResponse,
  ToolLoopConfig,
  AgenticResult
} from '../ai-provider.interface';
import { RateLimitError } from './adaptive-rate-limiter';
import { ConsoleLogger, Logger } from '../error-handling';

export interface FallbackProviderConfig {
  primaryProvider: AIProvider;
  fallbackProvider: AIProvider;
  logger?: Logger;
}

/**
 * Fallback Provider - automatically switches to fallback provider on rate limit errors
 * 
 * When primary provider (e.g., Azure OpenAI) returns 429 rate limit error,
 * automatically switches to fallback provider (e.g., AWS Bedrock) instead of retrying.
 * This prevents infinite retry loops and quota exhaustion issues.
 */
export class FallbackProvider implements AIProvider {
  private readonly primaryProvider: AIProvider;
  private readonly fallbackProvider: AIProvider;
  private readonly logger: Logger;
  private useFallback: boolean = false;

  constructor(config: FallbackProviderConfig) {
    this.primaryProvider = config.primaryProvider;
    this.fallbackProvider = config.fallbackProvider;
    this.logger = config.logger || new ConsoleLogger('FallbackProvider');
  }

  getProviderType(): string {
    return this.useFallback 
      ? `${this.fallbackProvider.getProviderType()} (fallback)`
      : this.primaryProvider.getProviderType();
  }

  getDefaultModel(): string {
    return this.useFallback
      ? this.fallbackProvider.getDefaultModel()
      : this.primaryProvider.getDefaultModel();
  }

  getModelName(): string {
    return this.useFallback
      ? this.fallbackProvider.getModelName()
      : this.primaryProvider.getModelName();
  }

  isInitialized(): boolean {
    return this.primaryProvider.isInitialized() && this.fallbackProvider.isInitialized();
  }

  async sendMessage(
    message: string,
    operation: string = 'chat',
    evaluationContext?: { user_intent?: string; interaction_id?: string }
  ): Promise<AIResponse> {
    try {
      // Try primary provider first
      if (!this.useFallback) {
        return await this.primaryProvider.sendMessage(message, operation, evaluationContext);
      }
    } catch (error) {
      // If rate limit error, switch to fallback provider
      if (error instanceof RateLimitError) {
        this.logger.warn('Primary provider rate limited, switching to fallback provider', {
          primaryProvider: this.primaryProvider.getProviderType(),
          fallbackProvider: this.fallbackProvider.getProviderType(),
          operation
        });
        this.useFallback = true;
      } else {
        // Re-throw non-rate-limit errors
        throw error;
      }
    }

    // Use fallback provider
    this.logger.info('Using fallback provider', {
      provider: this.fallbackProvider.getProviderType(),
      operation
    });
    return await this.fallbackProvider.sendMessage(message, operation, evaluationContext);
  }

  async toolLoop(config: ToolLoopConfig): Promise<AgenticResult> {
    try {
      // Try primary provider first
      if (!this.useFallback) {
        return await this.primaryProvider.toolLoop(config);
      }
    } catch (error) {
      // If rate limit error, switch to fallback provider
      if (error instanceof RateLimitError) {
        this.logger.warn('Primary provider rate limited during tool loop, switching to fallback', {
          primaryProvider: this.primaryProvider.getProviderType(),
          fallbackProvider: this.fallbackProvider.getProviderType(),
          operation: config.operation
        });
        this.useFallback = true;
      } else {
        // Re-throw non-rate-limit errors
        throw error;
      }
    }

    // Use fallback provider
    this.logger.info('Using fallback provider for tool loop', {
      provider: this.fallbackProvider.getProviderType(),
      operation: config.operation
    });
    return await this.fallbackProvider.toolLoop(config);
  }

  /**
   * Reset to use primary provider (useful for testing or after rate limits reset)
   */
  resetToPrimary(): void {
    this.useFallback = false;
    this.logger.info('Reset to primary provider', {
      provider: this.primaryProvider.getProviderType()
    });
  }
}
