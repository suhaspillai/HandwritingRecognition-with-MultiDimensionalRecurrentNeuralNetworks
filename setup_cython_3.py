#author : Suhas Pillai

'''
This script is used for creating .so file, which will be used by the cython  code.
'''

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

#setup(extensions =[Extension('lstm_layer_cy',['lstm_layer_cy.pyx'],extra_compile_args=["-g"],extra_link_args=["-g"])], ext_modules = cythonize(Extension('lstm_layer_cy',['lstm_layer_cy.pyx'],extra_compile_args=["-g"],extra_link_args=["-g"]),gdb_debug=True )
setup(extensions =[Extension('cython_mul_check_3',['cython_mul_check_3.pyx'],extra_compile_args=["-g"],extra_link_args=["-g"])], ext_modules = cythonize(Extension('cython_mul_check_3',['cython_mul_check_3.pyx'],extra_compile_args=["-g"],extra_link_args=["-g"],include_dirs=[numpy.get_include()]),gdb_debug=True )

      )
