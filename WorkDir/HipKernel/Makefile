CASE ?= gemm

compile:
	@echo Compiling $(CASE) for gfx1150...
	hipcc --offload-arch=gfx1150 $(CASE).hip -o $(CASE).exe

all:
	$(MAKE) clean
	$(MAKE) compile
	$(MAKE) run

clean:
	@echo Cleaning up...
	rm -f *.exe

run:
	@echo Running $(CASE)...
	./$(CASE).exe
