#!/usr/bin/env python3
"""Simple patch: replace 'raise ValueError' with 'pass #' in utils.py"""

import sys

site_packages = next(p for p in sys.path if 'site-packages' in p)
utils_file = site_packages + '/vllm/v1/worker/utils.py'

with open(utils_file, 'r') as f:
    lines = f.readlines()

# Find and patch the raise ValueError line
modified = False
for i, line in enumerate(lines):
    if 'raise ValueError(' in line and 'Free memory on device' in lines[i+1] if i+1 < len(lines) else False:
        # Replace this line and the next several lines with pass
        # Find the closing paren
        j = i
        paren_count = 0
        while j < len(lines):
            paren_count += lines[j].count('(') - lines[j].count(')')
            if paren_count <= 0:
                break
            j += 1
        
        # Replace from i to j with pass
        lines[i] = '        pass  # Memory check disabled for unified memory\n'
        # Remove the continuation lines
        for k in range(i+1, j+1):
            if k < len(lines):
                lines[k] = ''
        modified = True
        break

if modified:
    with open(utils_file, 'w') as f:
        f.writelines(lines)
    print('✓ Patched memory check (raise -> pass)')
else:
    # Try alternative: simple string replace
    with open(utils_file, 'r') as f:
        content = f.read()
    
    # Find the pattern and replace just the raise
    if 'raise ValueError(' in content and 'Free memory on device' in content:
        # More aggressive: replace the specific error message start
        old = 'raise ValueError('
        new = 'pass  # '
        content = content.replace(old, new, 1)  # Only first occurrence
        with open(utils_file, 'w') as f:
            f.write(content)
        print('✓ Patched memory check (simple replace)')
    else:
        print('✗ Pattern not found')
