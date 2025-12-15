import {
  AIProvider,
  AIResponse,
  ToolLoopConfig,
  AgenticResult
} from '../ai-provider.interface';
import { RateLimitError } from './adaptive-rate-limiter';
import { withAITracing } from '../tracing/ai-tracing';
import { ConsoleLogger, Logger } from '../error-handling';
import { CURRENT_MODELS } from '../model-config';
import * as https from 'https';

interface AzureProviderConfig {
  apiKey: string;
  endpoint: string;
  apiVersion: string;
  deployment: string;
  embeddingDeployment?: string;
  model?: string;
  logger?: Logger;
}

interface AzureUsage {
  promptTokens?: number;
  completionTokens?: number;
  totalTokens?: number;
}

interface AzureChoice {
  message: {
    content?: string;
  };
}

interface AzureChatResponse {
  choices: AzureChoice[];
  usage?: AzureUsage;
}

export class AzureOpenAIProvider implements AIProvider {
  private readonly apiKey: string;
  private readonly endpoint: string;
  private readonly apiVersion: string;
  private readonly deployment: string;
  private readonly embeddingDeployment?: string;
  private readonly model: string;
  private readonly logger: Logger;

  constructor(config: AzureProviderConfig) {
    this.apiKey = config.apiKey;
    this.endpoint = config.endpoint.replace(/\/+$/, '');
    this.apiVersion = config.apiVersion;
    this.deployment = config.deployment;
    this.embeddingDeployment = config.embeddingDeployment;
    this.model = config.model || CURRENT_MODELS.azure_openai;
    this.logger = config.logger || new ConsoleLogger('AzureOpenAIProvider');

    this.validateConfig();
  }

  private validateConfig(): void {
    const missing: string[] = [];
    if (!this.apiKey) missing.push('AZURE_OPENAI_API_KEY');
    if (!this.endpoint) missing.push('AZURE_OPENAI_ENDPOINT');
    if (!this.apiVersion) missing.push('AZURE_OPENAI_API_VERSION');
    if (!this.deployment) missing.push('AZURE_OPENAI_DEPLOYMENT');

    if (missing.length) {
      throw new Error(
        `Azure OpenAI provider missing configuration: ${missing.join(', ')}`
      );
    }
  }

  getProviderType(): string {
    return 'azure_openai';
  }

  getDefaultModel(): string {
    return CURRENT_MODELS.azure_openai;
  }

  getModelName(): string {
    return this.model;
  }

  isInitialized(): boolean {
    return true;
  }

  async sendMessage(
    message: string,
    _operation: string = 'azure-openai',
    _evaluationContext?: { user_intent?: string; interaction_id?: string }
  ): Promise<AIResponse> {
    return await withAITracing(
      {
        provider: this.getProviderType(),
        model: this.model,
        operation: 'chat'
      },
      async () => {
        const url = `${this.endpoint}/openai/deployments/${this.deployment}/chat/completions?api-version=${this.apiVersion}`;

        const body = {
          messages: [
            {
              role: 'user',
              content: message
            }
          ],
          temperature: 0.2
        };

        const response = await this.callAzure(url, body);
        const choice = response.choices?.[0];
        const text = choice?.message?.content ?? '';
        const usage = response.usage || {};

        return {
          content: text,
          usage: {
            input_tokens: usage.promptTokens ?? 0,
            output_tokens: usage.completionTokens ?? 0
          }
        };
      },
      (result: AIResponse) => ({
        inputTokens: result.usage.input_tokens,
        outputTokens: result.usage.output_tokens
      })
    );
  }

  async toolLoop(config: ToolLoopConfig): Promise<AgenticResult> {
    const combinedPrompt = `${config.systemPrompt}\n\n${config.userMessage}`;
    const response = await this.sendMessage(
      combinedPrompt,
      config.operation ?? 'azure-tool-loop',
      config.evaluationContext
    );

    return {
      finalMessage: response.content,
      iterations: 1,
      toolCallsExecuted: [],
      totalTokens: {
        input: response.usage.input_tokens,
        output: response.usage.output_tokens
      },
      status: 'success',
      completionReason: 'investigation_complete',
      modelVersion: this.getModelName()
    };
  }

  private async callAzure(url: string, body: any): Promise<AzureChatResponse> {
    const requestId = `azure-req-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    this.logger.info('Making Azure OpenAI API request', {
      requestId,
      timestamp: new Date().toISOString(),
      url: url.split('?')[0]
    });

    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      'api-key': this.apiKey,
      'Cache-Control': 'no-cache, no-store, must-revalidate',
      'Pragma': 'no-cache',
      'x-request-id': requestId
    };

    // Force fresh TCP connection for each request to prevent Azure from tracking
    // rate limits per connection. This mimics behavior of separate curl/script calls.
    const agent = new https.Agent({
      keepAlive: false,
      maxSockets: 1
    });

    const response = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
      // @ts-ignore - agent is valid but TypeScript doesn't recognize it in fetch options
      agent
    });

    if (response.status === 429) {
      const retryAfterHeader = response.headers.get('retry-after');
      const retryAfterMs = retryAfterHeader
        ? parseFloat(retryAfterHeader) * 1000
        : undefined;
      this.logger.warn('Azure OpenAI rate limit hit', {
        retryAfterHeader: retryAfterHeader || 'not provided',
        retryAfterMs: retryAfterMs || 'using backoff strategy',
        retryAfterSeconds: retryAfterHeader ? parseFloat(retryAfterHeader) : 'N/A'
      });
      throw new RateLimitError('Azure OpenAI rate limit exceeded', retryAfterMs);
    }

    if (!response.ok) {
      const text = await response.text();
      this.logger.error('Azure OpenAI request failed', undefined, {
        status: response.status,
        body: text
      });
      throw new Error(
        `Azure OpenAI request failed with status ${response.status}: ${text}`
      );
    }

    return (await response.json()) as AzureChatResponse;
  }
}
