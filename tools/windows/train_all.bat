@echo off
echo Training all variants...
python cli\train_standard.py --run-name standard
python cli\train_staged.py --run-name staged
python cli\train_curriculum.py --run-name curriculum
echo All training runs complete.
