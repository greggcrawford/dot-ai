import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { AzureOpenAIProvider } from '../../../src/core/providers/azure-openai-provider';
import { RateLimitError } from '../../../src/core/providers/adaptive-rate-limiter';

const baseConfig = {
  apiKey: 'key',
  endpoint: 'https://example.openai.azure.com',
  apiVersion: '2024-10-01-preview',
  deployment: 'gpt-4o',
};

describe('AzureOpenAIProvider', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn());
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('throws if required configuration is missing', () => {
    expect(
      () =>
        new AzureOpenAIProvider({
          ...baseConfig,
          apiKey: '',
        }),
    ).toThrow(/AZURE_OPENAI_API_KEY/);
  });

  it('sends chat completions and returns usage data', async () => {
    const provider = new AzureOpenAIProvider(baseConfig);

    (fetch as unknown as ReturnType<typeof vi.fn>).mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        choices: [{ message: { content: 'hello' } }],
        usage: { promptTokens: 10, completionTokens: 5 },
      }),
    });

    const response = await provider.sendMessage('ping');

    expect(fetch).toHaveBeenCalledWith(
      `${baseConfig.endpoint}/openai/deployments/${baseConfig.deployment}/chat/completions?api-version=${baseConfig.apiVersion}`,
      expect.objectContaining({
        method: 'POST',
      }),
    );
    expect(response.content).toBe('hello');
    expect(response.usage).toEqual({
      input_tokens: 10,
      output_tokens: 5,
    });
  });

  it('throws RateLimitError when Azure returns 429', async () => {
    const provider = new AzureOpenAIProvider(baseConfig);

    (fetch as unknown as ReturnType<typeof vi.fn>).mockResolvedValue({
      ok: false,
      status: 429,
      headers: new Map([['retry-after', '1']]),
    });

    await expect(provider.sendMessage('ping')).rejects.toBeInstanceOf(
      RateLimitError,
    );
  });

  it('throws descriptive error for non-OK responses', async () => {
    const provider = new AzureOpenAIProvider(baseConfig);

    (fetch as unknown as ReturnType<typeof vi.fn>).mockResolvedValue({
      ok: false,
      status: 500,
      text: async () => 'boom',
    });

    await expect(provider.sendMessage('ping')).rejects.toThrow(
      /status 500: boom/,
    );
  });
});
