from os import cpu_count
from setuptools import setup
from pathlib import Path
from setuptools import Extension
try:
    from Cython.Build import cythonize
    extm = cythonize([Extension('mnnpy._utils', 
                      ['mnnpy/_utils.pyx'],
                      extra_compile_args = ['-O2', '-ffast-math', '-march=native', '-fopenmp'],
                      extra_link_args=['-fopenmp'])])
except ImportError:
    print('Building with c.')
    extm = [Extension('mnnpy._utils', 
            ['mnnpy/_utils.c'],
            extra_compile_args = ['-O2', '-ffast-math', '-march=native', '-fopenmp'],
            extra_link_args=['-fopenmp'])]

req_path = Path('requirements.txt')
with req_path.open() as requirements:
    requires = [l.strip() for l in requirements]

setup(name='mnnpy',
      version='0.1.9.2',
      description='Mutual nearest neighbors correction in python.',
      long_description='Correcting batch effects in single-cell expression datasets using the mutual nearest neighbors method.',
      url='http://github.com/chriscainx/mnnpy',
      author='Chris Kang',
      author_email='kbxchrs@gmail.com',
      license='BSD 3',
      packages=['mnnpy'],
      install_requires=requires,
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering :: Bio-Informatics',
          'License :: OSI Approved :: BSD License',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
      ],
      python_requires='>=3.4',
      py_modules=['irlb', 'mnn', 'utils'],
      ext_modules=extm,
      zip_safe=False)
