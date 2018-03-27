all:
	python setup.py build

install:
	python setup.py install

check_format:
	$(call colorecho,"Checking formatting with flake8")
	@./project_tools/check_code_style_all
	@git diff --check

format:
	$(call colorecho,"Formatting with autopep8")
	@./project_tools/check_code_style_all --fix

.PHONY: format check_format all install
