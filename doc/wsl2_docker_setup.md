# wsl2_docker_setup note

## wsl2 GUI setup

1. Install GUI application software
	```
	sudo apt update -y
	sudo apt upgrade -y
	sudo apt install -y libgl1-mesa-dev xorg-dev xbitmaps x11-apps
	sudo apt -y install ubuntu-desktop
	```
1. Test
	```
	xeyes
	```  

## Docker environment setup

1. Make docker launch file and add options bellow
	```
	-e DISPLAY=$DISPLAY \ 
	-e WAYLAND_DISPLAY=$WAYLAND_DISPLAY \ 
	-e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \ 
	-v /tmp/.X11-unix:/tmp/.X11-uni \ 
	-v /mnt/wslg:/mnt/wslg \ 
	``` 

## Reference
- https://blog.mohyo.net/2022/02/11591/
