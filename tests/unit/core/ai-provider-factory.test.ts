import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  AIProviderFactory,
  createAIProvider,
} from '../../../src/core/ai-provider-factory';
import { NoOpAIProvider } from '../../../src/core/providers/noop-provider';

const originalEnv = { ...process.env };

function resetEnv() {
  process.env = { ...originalEnv };
}

describe('AIProviderFactory', () => {
  afterEach(() => {
    resetEnv();
    vi.resetModules();
  });

  it('returns NoOp provider when no keys configured', () => {
    process.env.AI_PROVIDER = 'openai';
    delete process.env.OPENAI_API_KEY;

    const provider = AIProviderFactory.createFromEnv();
    expect(provider).toBeInstanceOf(NoOpAIProvider);
  });

  it('creates Azure provider when env vars present', () => {
    process.env.AI_PROVIDER = 'azure_openai';
    process.env.AZURE_OPENAI_API_KEY = 'key';
    process.env.AZURE_OPENAI_ENDPOINT = 'https://example';
    process.env.AZURE_OPENAI_API_VERSION = '2024-10-01-preview';
    process.env.AZURE_OPENAI_DEPLOYMENT = 'gpt-4o';

    const provider = createAIProvider();

    expect(provider.getProviderType()).toBe('azure_openai');
  });
});
