#!/usr/bin/env python
"""
Generic python script.
"""
__author__ = "Alex Drlica-Wagner"
import subprocess

def test_bin():
    cmd = 'dsphsim > out1.txt'
    subprocess.check_call(cmd,shell=True)

    cmd = 'dsphsim -v --seed 0 --kinematics=Physical out2.txt'
    subprocess.check_call(cmd,shell=True)

if __name__ == "__main__":
    test_bin()
