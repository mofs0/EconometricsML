#!/usr/bin/env python
import os
import sys
import subprocess

# 明确设置 Python 路径
sys.path.insert(0, r'd:\Git\EconometricsML')
sys.path.insert(0, r'd:\anaconda3\envs\interview_prep\Lib\site-packages')

os.chdir(r'd:\Git\EconometricsML')

# 执行 nbconvert
result = subprocess.run([
    sys.executable, 
    '-m', 
    'jupyter',
    'nbconvert',
    '--to', 'notebook',
    '--execute',
    '--inplace',
    r'examples/econometrics/01_OLS_Angrist2008.ipynb'
], capture_output=False)

sys.exit(result.returncode)
