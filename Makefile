SHELL := /bin/sh
PY ?= python3

.PHONY: help test demo verify calibrate clean

help:
	@echo "make test      运行标准库单元测试"
	@echo "make demo      运行一次合成实验（含序贯查看）"
	@echo "make verify    断言统计输出"
	@echo "make calibrate 批量校准序贯设计（零/正/负效应与小样本）"

test:
	PYTHONPYCACHEPREFIX=/tmp/abkit-pycache $(PY) -m unittest discover -s tests -v

demo:
	PYTHONPATH=. $(PY) scripts/demo.py

verify:
	sh scripts/verify.sh

calibrate:
	PYTHONPATH=. $(PY) scripts/calibrate.py

clean:
	rm -rf /tmp/abkit-pycache
