#!/usr/bin/env python3
"""Patch vLLM FP8 MoE backend to fallback to TRITON instead of error"""

import sys

site_packages = next(p for p in sys.path if 'site-packages' in p)
fp8_oracle = site_packages + '/vllm/model_executor/layers/fused_moe/oracle/fp8.py'

with open(fp8_oracle, 'r') as f:
    content = f.read()

# Pattern for the NotImplementedError (exact match from file)
old_code = '''raise NotImplementedError(
                "Found VLLM_USE_FLASHINFER_MOE_FP8=1, but no "
                "FlashInfer FP8 MoE backend supports the configuration."
            )'''

# Fallback to TRITON
new_code = '''import warnings
                warnings.warn(
                    "FlashInfer FP8 MoE not available, falling back to TRITON backend",
                    RuntimeWarning
                )
                return (Fp8MoEBackend.TRITON, FusedMoEExpertV2)'''

if old_code in content:
    content = content.replace(old_code, new_code)
    with open(fp8_oracle, 'w') as f:
        f.write(content)
    print('✓ Patched FP8 MoE backend')
else:
    print('✗ FP8 pattern not found - checking if already patched...')
    if 'falling back to TRITON' in content:
        print('  Already patched!')
    else:
        print('  Could not find exact pattern - manual patch needed')
