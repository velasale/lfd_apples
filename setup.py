from setuptools import find_packages, setup

package_name = 'lfd_apples'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='alejo',
    maintainer_email='velasale@oregonstate.edu',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'listen_franka = lfd_apples.listen_franka:main',
            'free_drive_franka = lfd_apples.free_drive_franka:main',          
            'lfd_trial = lfd_apples.lfd_trial:main',         
        ],
    },
)
