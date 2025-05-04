from glob import glob
from setuptools import find_packages, setup

package_name = 'slam-mr'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (f'share/{package_name}/worlds', glob('worlds/*.world')),
        (f'share/{package_name}/urdf', glob('urdf/*.xacro')),
        (f'share/{package_name}/launch', glob('launch/*.launch.py')),
        (f'share/{package_name}/config', glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Thomas Kaminsky, Hammad Izhar',
    maintainer_email='hizhar@g.harvard.edu',
    description='CS2620 Distributed Computing Final Project',
    license='MIT',
    entry_points={
        'console_scripts': [
        ],
    },
)
