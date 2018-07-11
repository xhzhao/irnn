# build.py
import os
import platform
import sys
from distutils.core import setup

from torch.utils.ffi import create_extension

extra_compile_args = ['-std=c++11', '-fPIC']
iRNN_path = "../build"


if platform.system() == 'Darwin':
    lib_ext = ".dylib"
else:
    lib_ext = ".so"

headers = ['src/binding.h']

if "INTEL_RNN_PATH" in os.environ:
    iRNN_path = os.environ["INTEL_RNN_PATH"]
if not os.path.exists(os.path.join(iRNN_path, "libiRNN" + lib_ext)):
    print(("Could not find libiRNN.so in {}.\n"
           "Build iRNN and set INTEL_RNN_PATH to the location of"
           " libiRNN.so (default is '../build')").format(iRNN_path))
    sys.exit(1)
include_dirs = [os.path.realpath('../include')]

ffi = create_extension(
    name='irnn',
    language='c++',
    headers=headers,
    sources=['src/binding.cpp'],
    with_cuda=False,
    include_dirs=include_dirs,
    library_dirs=[os.path.realpath(iRNN_path)],
    runtime_library_dirs=[os.path.realpath(iRNN_path)],
    libraries=['iRNN'],
    extra_compile_args=extra_compile_args)
ffi = ffi.distutils_extension()
ffi.name = 'irnn_pytorch._irnn'
setup(
    name="irnn_pytorch",
    version="0.1",
    description="PyTorch wrapper for irnn",
    url="https://github.com/xhzhao/irnn",
    author="Xiaohui zhao, Qidu he, Shu zhang",
    author_email="Xiaohui.zhao@intel.com",
    license="Apache",
    packages=["irnn_pytorch"],
    ext_modules=[ffi],
)
