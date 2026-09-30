This project is developed on an AMD Ryzen™ AI Max+ 395 running Ubuntu 24.04. So here show the example to install the ROCm stack on this platform

The installation steps are based on https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installryz/native_linux/install-ryzen.html

```
Hardware: AMD Ryzen™ AI Max+ 395
OS: Ubuntu 24.04 + linux-oem-24.04c
ROCm: v7.2.+
```

## Prepare the system
For ROCm on Ryzen, it is required to operate on the 6.14-1018 OEM kernel or newer.

1. To install the kernel, please run the following command:

```
sudo apt update && sudo apt install linux-oem-24.04c
```

Once installation is complete, please reboot your system and ensure that you’ve booted into the correct kernel:

```
uname -r
```

Note: This returns a 6.14-1018 or newer based string.

3. Ensure that the system is up to date:

```
sudo apt upgrade -y
```

## Install AMD Unified Driver Package Repositories and Installer Script

```
sudo apt update
wget https://repo.radeon.com/amdgpu-install/7.2/ubuntu/noble/amdgpu-install_7.2.70200-1_all.deb
sudo apt install ./amdgpu-install_7.2.70200-1_all.deb
```

## Install AMD ROCm package

```
amdgpu-install -y --usecase=rocm --no-dkms
```

Set Groups permissions

```
groups
sudo usermod -a -G render,video $LOGNAME
sudo reboot
```

Post-install verification checks

```
rocminfo | grep gfx
```

