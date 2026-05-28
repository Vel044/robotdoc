# 一、这次改成配置文件方式
#
# QEMU-Pi5-rootfs.cfg 里放机器、内存、CPU 核数、磁盘、USB 设备。
# 命令行只保留 QEMU 8.2 不能稳定放进 readconfig 的入口参数：
#   -cpu
#   -nographic
#   -no-reboot
#   -kernel
#   -append
#
# 这仍然是 QEMU virt，不是完整 Raspberry Pi 5 模拟。


# 二、启动前检查

qemu-system-aarch64 --version

# 三、正常启动到命令行登录，带通用 USB host、hub 和真实 USB 透传
#
# 复制下面这一整段执行。
# 退出 QEMU：先按 Ctrl-a，再按 x。

sudo qemu-system-aarch64 \
  -readconfig /home/vel/QEMU-Pi5-rootfs.cfg \
  -cpu cortex-a76 \
  -nographic \
  -no-reboot \
  -kernel /home/vel/linux-rpi-6.12/arch/arm64/boot/Image \
  -append "console=ttyAMA0 root=/dev/vda2 rw rootwait systemd.unit=multi-user.target"


# 四、只读试跑，不修改 SD 镜像
# 复制下面这一整段执行。
# 加 -snapshot 后，guest 对磁盘的写入只进临时层，QEMU 退出后 img 不会变。

sudo qemu-system-aarch64 \
  -readconfig /home/vel/QEMU-Pi5-rootfs.cfg \
  -cpu cortex-a76 \
  -nographic \
  -no-reboot \
  -snapshot \
  -kernel /home/vel/linux-rpi-6.12/arch/arm64/boot/Image \
  -append "console=ttyAMA0 root=/dev/vda2 rw rootwait systemd.unit=multi-user.target"


# 五、维护模式：直接进 root shell
#
# 复制下面这一整段执行。
# 这段不启动 systemd，而是让 /bin/bash 直接作为 PID 1。
# 适合清理 rootfs、修配置、排查启动问题。

sudo qemu-system-aarch64 \
  -readconfig /home/vel/QEMU-Pi5-rootfs.cfg \
  -cpu cortex-a76 \
  -nographic \
  -no-reboot \
  -kernel /home/vel/linux-rpi-6.12/arch/arm64/boot/Image \
  -append "console=ttyAMA0 root=/dev/vda2 rw rootwait init=/bin/bash"


# 六、登录 guest 后建议检查
#
# 下面这些命令是在 QEMU 里的 Raspberry Pi rootfs 里执行，不是在宿主机执行。

systemctl --failed --no-pager

lsusb -t

ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null

dmesg | grep -Ei 'usb|serial|ttyUSB|ttyACM|ftdi|ch341'



# 七、真 UVC + CDC ACM 设备透传
#
# 这些透传设备已经写进 QEMU-Pi5-rootfs.cfg：
#   hostbus=1,hostport=4.1 -> guest xHCI port 6，UVC camera 1
#   hostbus=1,hostport=4.2 -> guest 第二个 xHCI port 1，UVC camera 2
#   hostbus=1,hostport=3.1 -> guest hub port 3，CDC ACM
#   hostbus=1,hostport=3.2 -> guest hub port 4，CDC ACM
#
# 目标：让 guest 的 USB 树尽量接近 01_USB拓扑.md：
#   xhci_hcd root hub
#     -> QEMU USB hub
#       -> Port 3: CDC ACM serial，Driver=cdc_acm
#       -> Port 4: CDC ACM serial，Driver=cdc_acm
#   xhci_hcd root hub
#     -> xHCI 高速端口：UVC camera 1 / camera 2，Driver=uvcvideo
# 当前配置把第二路 UVC 放到第二个 QEMU xHCI 控制器，减轻双路视频流互相阻塞。
#
# 注意：QEMU 自带 usb-hub 是 12M full-speed hub。
# 480M UVC camera 不能挂到 QEMU hub 下游，否则会报 speed mismatch。
# 所以当前阶段 UVC 先直接挂 xHCI 高速端口；这是和真机 USB 树的已知差异。

lsusb

ls -l /dev/video* /dev/media* /dev/ttyACM* /dev/ttyUSB* 2>/dev/null

# 当前宿主机这次检测到两个原始 USB 2.0 Camera：
#   hostbus=1,hostport=4.1 -> /dev/video0 / /dev/video1
#   hostbus=1,hostport=4.2 -> /dev/video4 / /dev/video5
# 当前宿主机这次检测到两个 QinHeng USB Single Serial：
#   hostbus=1,hostport=3.1 -> /dev/ttyACM0
#   hostbus=1,hostport=3.2 -> /dev/ttyACM1
# hostaddr 每次插拔都会变；hostport 按物理端口匹配，更稳定。
# 只要设备插回同一个 VMware USB hub 的同一个孔，就不用改 QEMU 命令。
# 宿主机 USB bus 节点是 root:root，普通用户会 failed to open host usb device。
# 启动前先退出正在运行的 QEMU，否则 SD 镜像会有写锁。

# 带真实 UVC + CDC ACM 透传
sudo qemu-system-aarch64 \
  -readconfig /home/vel/QEMU-Pi5-rootfs.cfg \
  -cpu cortex-a76 \
  -nographic \
  -no-reboot \
  -kernel /home/vel/linux-rpi-6.12/arch/arm64/boot/Image \
  -append "console=ttyAMA0 root=/dev/vda2 rw rootwait systemd.unit=multi-user.target"


sudo poweroff


# 八、guest 里验证 UVC 和 CDC ACM
#
# 下面这些命令是在 QEMU 里的 Raspberry Pi rootfs 里执行。


lsusb -t

ls -l /dev/video* /dev/media* /dev/ttyACM* /dev/ttyUSB* 2>/dev/null

dmesg | grep -Ei 'uvc|video|cdc_acm|ttyACM|usb 1-|usbcore|xhci'
