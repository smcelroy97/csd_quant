from __future__ import annotations
import re
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat


def _strip_comments(line: str) -> str:
    # remove MATLAB comments (% ...), but keep % inside quotes (simple heuristic)
    out = []
    in_str = False
    i = 0
    while i < len(line):
        ch = line[i]
        if ch == "'":
            in_str = not in_str
            out.append(ch)
        elif ch == "%" and not in_str:
            break
        else:
            out.append(ch)
        i += 1
    return "".join(out).strip()


'''
data_dir = Path('crcns_data/ac-2/Anesthetized_wholecell_and_LFP_data')
for child in data_dir.iterdir():
    for file in child.iterdir():
        metadata = {}
        print(str(file))
        if 'all_sweeps_both_channels.mat' in str(file):
            data_path = file
            combo_dat = loadmat(file)
            print('loaded')
        if 'cell_specific_parameters.m' in str(file):
            m_path = file
            for raw in Path(m_path).read_text(errors="ignore").splitlines():
                line = _strip_comments(raw)
                print(line)
                if not line or line.startswith('%'):
                    continue
                match = re.match(r'(\w+)\s*=\s*(\d+)\s*;', line)
                if match:
                    key = match.group(1)
                    value = int(match.group(2))
                    metadata[key] = value
                    print(metadata)
        else:
            continue
        combo_dat['metadata'] = metadata
'''
