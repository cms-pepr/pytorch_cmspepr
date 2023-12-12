import os
import os.path as osp
from setuptools import setup, find_packages
from textwrap import dedent

import torch
from torch.utils.cpp_extension import BuildExtension
from torch.utils.cpp_extension import CppExtension, CUDAExtension, CUDA_HOME

CUDA_AVAILABLE = torch.cuda.is_available() and CUDA_HOME is not None

DO_CPU = True
DO_CUDA = CUDA_AVAILABLE

if os.getenv('FORCE_ONLY_CPU', '0') == '1':
    print('FORCE_ONLY_CPU: Only compiling CPU extensions')
    DO_CPU = True
    DO_CUDA = False
elif os.getenv('FORCE_ONLY_CUDA', '0') == '1':
    print('FORCE_ONLY_CUDA: Only compiling CUDA extensions')
    DO_CPU = False
    DO_CUDA = True
elif os.getenv('FORCE_CUDA', '0') == '1':
    print('FORCE_CUDA: Forcing compilation of CUDA extensions')
    if not CUDA_AVAILABLE: print(f'{CUDA_AVAILABLE=}, high chance of failure')
    DO_CUDA = True

BUILD_DOCS = os.getenv('BUILD_DOCS', '0') == '1'


# Define extensions
extensions_dir = osp.join(osp.dirname(osp.abspath(__file__)), 'extensions')
cpu_kwargs = dict(
    include_dirs=[extensions_dir],
    extra_compile_args={'cxx': ['-O2']},
    extra_link_args=['-s']
    )
extensions = [
    CppExtension('select_knn_cpu', ['extensions/select_knn_cpu.cpp'], **cpu_kwargs),
    ]
if DO_CUDA:
    cuda_kwargs = dict(
        include_dirs=[extensions_dir],
        extra_compile_args={'cxx': ['-O2'], 'nvcc': ['--expt-relaxed-constexpr', '-O2']},
        extra_link_args=['-s']
        )
    extensions.extend([
        CUDAExtension(
            'select_knn_cuda',
            ['extensions/select_knn_cuda.cpp', 'extensions/select_knn_cuda_kernel.cu'],
            **cuda_kwargs
            ),
        ])


# Print extensions
def repr_ext(ext):
    """
    Debug print for an extension
    """
    return dedent(f"""\
        {ext.name}
          sources: {', '.join(ext.sources)}
          extra_compile_args: {ext.extra_compile_args}
          extra_link_args: {ext.extra_link_args}
        """)

print('\n---------------------\nExtensions:')
for ext in extensions: print(repr_ext(ext))
print('---------------------')


# Setup call
tests_require = ['pytest', 'pytest-cov', 'scipy']
setup(
    name='torch_cmspepr',
    version='1.0.0',
    author='Lindsey Gray <Lindsey.Gray@cern.ch>, Jan Kieseler <jan.kieseler@cern.ch>, Thomas Klijnsma <thomasklijnsma@gmail.com>',
    author_email='Lindsey.Gray@cern.ch',
    url='',
    description=('PyTorch Extension Library for HGCAL Specific knn optimizations'),
    keywords=[
        'pytorch',
        'knn',
        'geometric-deep-learning',
        'graph-neural-networks',
        'cluster-algorithms',
    ],
    license='MIT',
    python_requires='>=3.6',
    install_requires=[],
    setup_requires=['pytest-runner'],
    tests_require=tests_require,
    extras_require={'test': tests_require},
    ext_modules=extensions if not BUILD_DOCS else [],
    cmdclass={
        'build_ext':
        BuildExtension.with_options(no_python_abi_suffix=True, use_ninja=False)
    },
    packages=find_packages(),
)
