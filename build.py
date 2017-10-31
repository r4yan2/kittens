from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
    name="kittens engine",
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("kittens", ['kittens.py']), Extension("database", ['database.py']), Extension("helper", ['helper.py']), Extension("recommender", ['recommender.py'])]
)