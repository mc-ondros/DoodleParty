/**
 * Smoke test for frontend CI pipeline.
 * Ensures test infrastructure works and basic imports succeed.
 */

import React from 'react';
import { describe, it, expect } from 'vitest';

describe('Frontend Smoke Tests', () => {
  it('should pass basic assertion', () => {
    expect(true).toBe(true);
  });

  it('should have React available', () => {
    expect(React).toBeDefined();
  });

  it('should support JSX', () => {
    const element = React.createElement('div', null, 'test');
    expect(element).toBeDefined();
    expect(element.type).toBe('div');
  });
});
