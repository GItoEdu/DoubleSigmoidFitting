@echo off
cd /d "%~dp0"
i:\Ubuntu\VSCode\DoubleSigmoidFitting\venv\Scripts\python.exe fit_double_sigmoid_auto.py "%~1"
pause