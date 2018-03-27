try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup

config = {
    'description': 'Trajectory generation for TORQ Quadrotors',
    'author': 'Gene Merewether',
    'url': 'https://github.com/genemerewether/traj',
    'download_url': 'https://github.com/genemerewether/traj',
    'author_email': 'genemerewether@gmail.com',
    'version': '0.1',
    'setup_requires': ['numpy>=1.12.0'],
    'install_requires': ['setuptools', 'numpy>=1.12.0', 'scipy>=0.19.0',
                         'pyyaml', 'future', 'rdp', 'lxml','transforms3d', 'cvxopt'],
    'packages': find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    'name': 'traj'
}

setup(**config)
