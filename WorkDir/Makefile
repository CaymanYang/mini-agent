compile:
	@echo Compiling hip_gemm for gfx1150...
	hipcc --offload-arch=gfx1150 hip_gemm.hip -o hip_gemm.exe

all:
	$(MAKE) clean
	$(MAKE) compile
	$(MAKE) run

clean:
	@echo Cleaning up...
	rm -f hip_gemm.exe

run:
	@echo Running hip_gemm...
	.\hip_gemm.exe