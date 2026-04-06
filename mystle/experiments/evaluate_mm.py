#!/usr/bin/env python3
"""
Step 5 多模态评测（兼容入口）。

等价于::

    python mystle/experiments/evaluate.py mm [OPTIONS]

完整说明见 ``mystle/experiments/evaluate.py`` 模块文档字符串。
"""

from __future__ import annotations

import sys


def main() -> None:
    sys.argv = [sys.argv[0], "mm", *sys.argv[1:]]
    from mystle.experiments.evaluate import main as _eval_main

    _eval_main()


if __name__ == "__main__":
    main()
