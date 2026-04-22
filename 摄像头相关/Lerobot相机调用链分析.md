# Lerobot 相机调用链分析（推理路径）

**实际硬件配置（两路摄像头）：**

```
--robot.cameras="{'handeye': {'index_or_path': 0, 'width': 640, 'height': 360},
                  'fixed':   {'index_or_path': 2, 'width': 640, 'height': 360}}"
```

- `handeye`：手眼相机，`/dev/video0`，640×360
- `fixed`：固定视角相机，`/dev/video2`，640×360
- 两路相机各自启动独立后台线程，调用链结构相同，以下以单路为例说明

---

## 1. 完整调用链概览

```
【主线程】
record.py / record_loop()
    └─ robot.get_observation()
       robots/so101_follower/so101_follower.py / SO101Follower.get_observation()
           └─ cam.async_read(timeout_ms=3000)
              cameras/opencv/camera_opencv.py / OpenCVCamera.async_read()
                  ├─ [线程未启动时] self._start_read_thread()
                  │  cameras/opencv/camera_opencv.py / OpenCVCamera._start_read_thread()
                  │      └─ Thread(target=self._read_loop, daemon=True).start()
                  │         → 后台线程启动，主线程继续
                  └─ self.new_frame_event.wait(timeout=3.0)
                     → 阻塞等待后台线程通知新帧到达
                     → 超时抛出 TimeoutError

【后台采集线程，循环运行】
cameras/opencv/camera_opencv.py / OpenCVCamera._read_loop()
    └─ self.read()
       cameras/opencv/camera_opencv.py / OpenCVCamera.read()
           └─ self.videocapture.read()          (Python → OpenCV C++)
              ↓↓↓  以下进入 OpenCV 内部，见《OpenCV视频流内部实现.md》
           └─ self._postprocess_image(frame)    (BGR→RGB + 旋转)
       → with frame_lock: self.latest_frame = color_image
       → self.new_frame_event.set()   → 通知主线程新帧已到达
```

---

## 2. 主线程：观测请求路径

### 2.1 `record.py / record_loop()` — 观测阶段入口

**源码路径**：`lerobot/src/lerobot/record.py:676-681`

```python
obs_hw_start_t = time.perf_counter()
obs = robot.get_observation()   # 阻塞调用，返回关节位置 + 相机图像
obs_hw_end_t = time.perf_counter()
```

`robot` 是 `Robot` 抽象类实例（此处为 `SO101Follower`），`get_observation()` 返回：
- `"{motor_name}.pos"` → `float`：6个电机位置
- `"{cam_key}"` → `np.ndarray`：每个相机的最新图像

---

### 2.2 `so101_follower.py / SO101Follower.get_observation()` — 电机 + 相机并列读取

**源码路径**：`lerobot/src/lerobot/robots/so101_follower/so101_follower.py:601-658`

```python
def get_observation(self) -> dict[str, Any]:
    if not self.is_connected:
        raise DeviceNotConnectedError(f"{self} is not connected.")

    # 读取电机位置（阻塞，~1-2ms）
    obs_dict = self.bus.sync_read("Present_Position")
    obs_dict = {f"{motor}.pos": val for motor, val in obs_dict.items()}

    # 读取每个相机最新帧（等待后台线程通知）
    for cam_key, cam in self.cameras.items():
        obs_dict[cam_key] = cam.async_read(timeout_ms=3000)

    return obs_dict
```

---

### 2.3 `camera_opencv.py / OpenCVCamera.async_read()` — 等待新帧

**源码路径**：`lerobot/src/lerobot/cameras/opencv/camera_opencv.py:754-825`

```python
def async_read(self, timeout_ms: float = 200) -> np.ndarray:
    if not self.is_connected:
        raise DeviceNotConnectedError(f"{self} is not connected.")

    # 后台线程未启动时自动启动
    if self.thread is None or not self.thread.is_alive():
        self._start_read_thread()

    # 阻塞等待后台线程写入新帧
    if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
        raise TimeoutError(f"Timed out waiting for frame from camera {self} after {timeout_ms} ms.")

    # 线程安全取帧
    with self.frame_lock:
        frame = self.latest_frame
        self.new_frame_event.clear()

    return frame
```

**`frame_lock` 互斥锁说明：**

`latest_frame` 被两个线程同时访问：后台采集线程 `_read_loop` 持续**写入**，主线程 `async_read` 负责**读取**。没有锁保护时，可能读到写了一半的损坏帧。

`with self.frame_lock` 块内三步必须原子执行：

1. **申请锁** — 后台线程此时无法写入 `latest_frame`
2. **读取帧引用** — `frame = self.latest_frame`，拿到完整帧
3. **清除新帧信号** — `self.new_frame_event.clear()`，必须在锁内完成

第 3 步放在锁内的原因：若先释放锁再 `clear()`，后台线程可能在这个间隙写入新帧并 `set()` event，随后的 `clear()` 会把这个新信号也清除，导致该帧永远不触发通知（帧丢失）。

---


## 3. 后台采集线程：帧采集路径

### 3.1 `camera_opencv.py / OpenCVCamera._start_read_thread()` — 启动后台线程

**源码路径**：`lerobot/src/lerobot/cameras/opencv/camera_opencv.py:706-731`

```python
def _start_read_thread(self) -> None:
    if self.thread is not None and self.thread.is_alive():
        self.thread.join(timeout=0.1)
    if self.stop_event is not None:
        self.stop_event.set()

    self.stop_event = Event()
    self.thread = Thread(target=self._read_loop, args=(), name=f"{self}_read_loop")
    self.thread.daemon = True   # 守护线程，随主线程退出自动结束
    self.thread.start()
```

---

### 3.2 `camera_opencv.py / OpenCVCamera._read_loop()` — 循环采集

**源码路径**：`lerobot/src/lerobot/cameras/opencv/camera_opencv.py:667-704`

```python
def _read_loop(self):
    while not self.stop_event.is_set():
        try:
            color_image = self.read()           # 阻塞读取一帧（等相机硬件，~33ms@30fps）

            with self.frame_lock:
                self.latest_frame = color_image # 线程安全写入最新帧
            self.new_frame_event.set()          # 通知主线程新帧已到达

        except DeviceNotConnectedError:
            break
        except Exception as e:
            logger.warning(f"Error reading frame in background thread for {self}: {e}")
```

---

### 3.3 `camera_opencv.py / OpenCVCamera.read()` — 同步读一帧

**源码路径**：`lerobot/src/lerobot/cameras/opencv/camera_opencv.py:539-597`

```python
def read(self, color_mode: ColorMode | None = None) -> np.ndarray:
    if not self.is_connected:
        raise DeviceNotConnectedError(f"{self} is not connected.")

    ret, frame = self.videocapture.read()   # 阻塞，等待 V4L2 下一帧（~33ms@30fps）

    if not ret or frame is None:
        raise RuntimeError(f"{self} read failed (status={ret}).")

    processed_frame = self._postprocess_image(frame, color_mode)  # BGR→RGB + 旋转
    return processed_frame
```

`self.videocapture.read()` 返回的 `frame` 是一个三维 numpy 数组：
- **shape**：`(capture_height, capture_width, 3)`，例如 `(480, 640, 3)`
- **dtype**：`uint8`，每个像素通道值 0–255
- **通道顺序**：**BGR**（OpenCV 惯例，蓝-绿-红）

这个 BGR 数组是 OpenCV 内部 grab→retrieve→MJPEG解码 后的直接输出，随后交给 `_postprocess_image` 做颜色转换。

**OpenCV 内部入口**：`self.videocapture.read()` 经 Python 绑定进入 C++ `VideoCapture::read()`（`cap.cpp:551`），立即拆分为 `grab()` + `retrieve()` 两阶段，详见《OpenCV视频流内部实现.md》§1。

---

### 3.4 `camera_opencv.py / OpenCVCamera._postprocess_image()` — 颜色转换与旋转

**源码路径**：`lerobot/src/lerobot/cameras/opencv/camera_opencv.py:599-665`

```python
def _postprocess_image(self, image: np.ndarray, color_mode: ColorMode | None = None) -> np.ndarray:
    # 确定颜色模式（参数优先，其次实例配置）
    requested_color_mode = self.color_mode if color_mode is None else color_mode
 
    # BGR → RGB 转换（需要）
    processed_image = image
    if requested_color_mode == ColorMode.RGB:
        processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 旋转（没配置，跳过）
    if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]:
        processed_image = cv2.rotate(processed_image, self.rotation)

    return processed_image
```

**默认配置下的行为：**

- **BGR→RGB 转换**：默认 `color_mode=ColorMode.RGB`，**每帧都会执行**，约 0.1–0.5ms @ 640×480。返回给上层的 `np.ndarray` 始终是 RGB 格式，shape `(H, W, 3)`，dtype `uint8`。
- **旋转**：默认 `rotation=Cv2Rotation.NO_ROTATION`，`self.rotation` 为 `None`，**跳过**，`cv2.rotate` 不调用。
