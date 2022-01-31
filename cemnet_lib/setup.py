#python3 setup.py install
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

libname = "cemnet_lib"

setup(
    name='cemnet_lib',
    ext_modules=[
        CUDAExtension("cemnet_lib_cuda", [
            "src/cemnet_lib_api.cpp",
            "src/ops/ops_cuda.cpp",
            "src/ops/ops_cuda_kernel.cu",
        ],
        extra_compile_args={'cxx': ['-O2', '-I{}'.format('{}/include'.format(libname))],'nvcc': ['-O2']})
    ],
    cmdclass={'build_ext': BuildExtension}
)