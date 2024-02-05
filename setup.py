from setuptools import setup

setup_requires = []

install_requires = ["numpy"]

setup(
    name="imaml",
    version="0.0.0",
    description="implicit maml",
    author="Hirokazu Ishida",
    author_email="h-ishida@jsk.imi.i.u-tokyo.ac.jp",
    install_requires=install_requires,
    package_data={"imaml": ["py.typed"]},
)
