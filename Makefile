SHELL := /bin/sh
PY ?= python3

.PHONY: help test demo verify clean

help:
	@echo "make test     运行标准库单元测试"
	@echo "make demo     运行一次合成实验"
	@echo "make verify   断言统计输出"

test:
	PYTHONPYCACHEPREFIX=/tmp/abkit-pycache $(PY) -m unittest discover -s tests -v

demo:
	PYTHONPATH=. $(PY) scripts/demo.py

verify:
	sh scripts/verify.sh

clean:
	rm -rf /tmp/abkit-pycache
