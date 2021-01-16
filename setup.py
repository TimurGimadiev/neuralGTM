# -*- coding: utf-8 -*-
#
#  Copyright 2021 Timur Gimadiev <timur.gimadiev@gmail.com>
#  This file is part of neuralGTM.
#
#  neuralGTM is free software; you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation; either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with this program; if not, see <https://www.gnu.org/licenses/>.
#
from setuptools import setup, find_packages
from pathlib import Path

setup(
    name='neuralGTM',
    version='0.0.1',
    description='A sklearn compatible implementation of Generative Topographic Mapping.',
    long_description=(Path(__file__).parent / 'README.md').read_text(),
    author='Timur Gimadiev',
    author_email='timur.gimadiev@gmail.com',
    install_requires=['numpy>=1.18', 'scipy', 'scikit-learn', 'torch'],
    url='https://github.com/TimurGimadiev/neuralGTM.git',
    license='LGPLv3',
    packages=find_packages(exclude=('tests', 'docs')),
    zip_safe=True,
    classifiers=['Environment :: Plugins',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
                 'Operating System :: OS Independent',
                 'Programming Language :: Python',
                 'Programming Language :: Python :: 3 :: Only',
                 'Programming Language :: Python :: 3.8',
                 'Topic :: Scientific/Engineering',
                 'Topic :: Scientific/Engineering :: Chemistry',
                 'Topic :: Scientific/Engineering :: Information Analysis',
                 'Topic :: Software Development',
                 'Topic :: Software Development :: Libraries',
                 'Topic :: Software Development :: Libraries :: Python Modules']
)
