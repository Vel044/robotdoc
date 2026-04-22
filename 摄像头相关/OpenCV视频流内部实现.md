# OpenCV 视频流内部实现（V4L2 层）

**实际硬件配置（两路摄像头）：**

```
--robot.cameras="{'handeye': {'index_or_path': 0, 'width': 640, 'height': 360},
                  'fixed':   {'index_or_path': 2, 'width': 640, 'height': 360}}"
```

- `handeye`：`/dev/video0`；`fixed`：`/dev/video2`；各自对应独立的 `cv2.VideoCapture` 实例
- 以下描述单路实例的 C++ 内部路径，两路结构相同

---

本文档描述 `self.videocapture.read()` 以下的 OpenCV C++ 实现，即从 lerobot 调用 Python 绑定开始，经 V4L2 ioctl 到内核的完整路径。lerobot 侧的调用栈见《Lerobot相机调用链分析.md》。

**调用入口**：lerobot 的 `self.videocapture.read()`（Python）经绑定层进入 C++ `VideoCapture::read()`（`cap.cpp:551`），这是本文档描述的起点，见 §1。

**返回约定**：`VideoCapture::read()` 执行完毕后，向 Python 层返回 `(bool, np.ndarray)`，其中 `np.ndarray` 是 shape `(H, W, 3)`、dtype `uint8`、通道顺序 **BGR** 的三维数组，由 MJPEG 解码直接填充，lerobot 的 `_postprocess_image` 负责将其转为 RGB。

---

## 完整调用链

```
Python: self.videocapture.read()
    │  （Python binding → C++）
    ▼
VideoCapture::read()                                    opencv/modules/videoio/src/cap.cpp:551
    │
    │  ┌─────────────────────────────────────────────────────────────────────────────────┐
    ├─ │ grab()   【硬件出队，不解码】                      cap.cpp:524                  │
    │  │  只做一件事：VIDIOC_DQBUF 把内核 DMA 缓冲区的指针                               │
    │  │  "领"到用户态。数据还在 mmap 的原始格式（MJPEG 压缩数据），CPU 几乎不消耗。       │
    │  │  设计意图：多路摄像头时可以对所有摄像头连续调 grab()，                           │
    │  │  让所有摄像头的帧在时间上对齐，再统一 retrieve() 解码。                          │
    │  └─────────────────────────────────────────────────────────────────────────────────┘
    │    └─ icap->grabFrame()                 （虚函数，V4L2 后端）
    │         └─ CvCaptureCAM_V4L::grabFrame()          opencv/modules/videoio/src/cap_v4l.cpp:1204
    │              └─ read_frame_v4l2()                 opencv/modules/videoio/src/cap_v4l.cpp:1015
    │                   └─ tryIoctl(VIDIOC_DQBUF)       opencv/modules/videoio/src/cap_v4l.cpp:1114
    │                        ├─ ioctl(VIDIOC_DQBUF)   ← 出队已填充帧，成功则返回
    │                        └─ [EAGAIN] select()      ← 阻塞等待新帧 ~33ms@30fps
    │
    │  ┌─────────────────────────────────────────────────────────────────────────────────┐
    └─ │ retrieve()   【解码 + 拷贝，CPU 重头】              cap.cpp:535                  │
       │  拿到 grab() 留在 mmap 里的压缩数据，做格式转换：                               │
       │  MJPEG → libjpeg 解码 → BGR cv::Mat → np.ndarray。                             │
       │  这一步是整条管线 CPU 最重的操作（10-30ms@1080p）。                             │
       │  lerobot 单路摄像头：grab+retrieve 顺序执行，分离无实际意义。                    │
       └─────────────────────────────────────────────────────────────────────────────────┘
         └─ icap->retrieveFrame()             （虚函数，V4L2 后端）
              └─ CvCaptureCAM_V4L::retrieveFrame()      opencv/modules/videoio/src/cap_v4l.cpp:2262
                   ├─ convertToRgb()                    opencv/modules/videoio/src/cap_v4l.cpp:1553
                   │    └─ [MJPEG] cv::imdecode()     ← libjpeg 解码，10-30ms@1080p
                   ├─ tryIoctl(VIDIOC_QBUF)           ← 归还缓冲区，供下次 grabFrame 使用
                   └─ frame.copyTo(ret)               ← cv::Mat → np.ndarray（BGR）
```

---

## 1. VideoCapture::read 的两阶段模式

OpenCV 使用 **grab/retrieve 分离模式**，但 `read()` 会自动组合：

```cpp
// opencv/modules/videoio/src/cap.cpp:551
bool VideoCapture::read(OutputArray image)
{
    if (grab()) {
        retrieve(image);
    } else {
        image.release();
    }
    return !image.empty();
}
```

lerobot 每次 `read()` 触发的是完整的 `grab + retrieve`，不是只走 grabFrame。

---

## 2. grabFrame: 从内核 DMA 缓冲区取数据

`grabFrame()` 的核心任务是：从内核的 DMA 缓冲区队列中取出一个已填充数据的缓冲区。

```cpp
// opencv/modules/videoio/src/cap_v4l.cpp:1204
bool CvCaptureCAM_V4L::grabFrame()
{
    // 正常 read() = grab+retrieve，到这里 bufferIndex 必然是 -1，此块不执行
    // 仅在"连续 grab 不 retrieve"时触发（多路同步接口跳帧场景）：
    if (bufferIndex >= 0) {
        if (!tryIoctl(VIDIOC_QBUF, &buffers[bufferIndex].buffer)) {
            CV_LOG_DEBUG(NULL, "VIDEOIO(V4L2:...): failed VIDIOC_QBUF");
        }
    }

    return read_frame_v4l2();  // 实际出队（VIDIOC_DQBUF）
}
```

### read_frame_v4l2: 核心出队操作

```cpp
// opencv/modules/videoio/src/cap_v4l.cpp:1015
bool CvCaptureCAM_V4L::read_frame_v4l2()
{
    v4l2_buffer buf = v4l2_buffer();   // 清零 v4l2_buffer，用于向内核传递/接收缓冲区信息

    buf.type   = type;                 // V4L2_BUF_TYPE_VIDEO_CAPTURE（单平面，lerobot MJPEG 走这里）
    buf.memory = V4L2_MEMORY_MMAP;    // 告诉内核：缓冲区是 mmap 方式分配的

    // tryIoctl(VIDIOC_DQBUF, &buf)：
    //   参数1 VIDIOC_DQBUF = 0xC0565611，告诉内核"我要取出一个已填充的缓冲区"
    //   参数2 &buf = v4l2_buffer*，输入 buf.index（要取哪个缓冲区），内核填充 buf.bytesused/timestamp
    //   返回 true = 出队成功，buf 里有完整的帧元数据；false = 失败（EAGAIN/EBUSY/EIO）
    //   EAGAIN 时 tryIoctl 内部会 select() 阻塞等新帧就绪后自动重试
    while (!tryIoctl(VIDIOC_DQBUF, &buf)) {
        int err = errno;
        // EIO + 缓冲区既不在队列也没完成：驱动内部状态异常，尝试重新入队恢复
        if (err == EIO && !(buf.flags & (V4L2_BUF_FLAG_QUEUED | V4L2_BUF_FLAG_DONE))) {
            if (!tryIoctl(VIDIOC_QBUF, &buf))  // VIDIOC_QBUF：把这块坏缓冲区重新交还驱动
                return false;
            continue;                           // 重试 DQBUF
        }
        returnFrame = false;   // 其他错误（摄像头拔出等），标记本帧无效
        return false;
    }
    // 到这里：buf.index 是出队缓冲区编号，buf.bytesused 是有效数据字节数（MJPEG 压缩大小）

    CV_Assert(buf.index < req.count);  // 防御性检查：index 不能超过申请的缓冲区总数

    // DQBUF 不拷贝像素数据，像素已在 mmap 内存里。这里只保存元数据：
    //   buf.index（缓冲区编号）、buf.bytesused（MJPEG 压缩字节数）、
    //   buf.timestamp（帧时间戳）、buf.flags（状态标志）
    buffers[buf.index].buffer = buf;
    bufferIndex = buf.index;
    timestamp = buf.timestamp;  // 记录帧时间戳（硬件填入，精度优于 gettimeofday）
    return true;                // 出队成功，mmap 缓冲区数据已就绪，等待 retrieveFrame() 解码
}
```

### tryIoctl: ioctl 重试包装 + select 等待

**VIDIOC_DQBUF 做了什么**（整个管线最核心的操作）：

内核和用户态通过 mmap 共享同一块物理内存，缓冲区始终在这块内存里，不会移动。DQBUF 改变的是"谁能访问"：

```
入队 QBUF  → 用户态交出缓冲区，驱动可以让摄像头 DMA 往里写数据
DMA 写入   → 摄像头硬件把一帧 MJPEG 直接写到物理内存，CPU 不参与
出队 DQBUF → 内核把缓冲区"所有权"交回用户态，用户态可以通过 mmap 地址读数据
```

DQBUF 不拷贝任何像素数据，只是一个访问权限转移。成功后 `buf` 里被内核填充了元数据：
- `buf.index`：第几号缓冲区（用于定位 mmap 地址）
- `buf.bytesused`：MJPEG 压缩了多少字节
- `buf.timestamp`：帧时间戳（硬件填入）

`tryIoctl` 是 ioctl 的重试包装：DQBUF 时如果帧还没到（EAGAIN），内部调 `select()` 阻塞等摄像头就绪后自动重试。

**调用方式**：`tryIoctl(VIDIOC_DQBUF, &buf)`，后两个参数用默认值：
- `ioctlCode` = `VIDIOC_DQBUF`（0xC0565611）
- `parameter` = `&buf`（`v4l2_buffer*`）
- `failIfBusy` = 默认 `true`
- `attempts` = 默认 `10`

```cpp
// opencv/modules/videoio/src/cap_v4l.cpp:1114
bool CvCaptureCAM_V4L::tryIoctl(unsigned long ioctlCode,    // ioctl 命令码，如 VIDIOC_DQBUF=0xC0565611
                                 void *parameter,            // v4l2_buffer*，输入缓冲区编号，内核填充元数据
                                 bool failIfBusy,            // EBUSY 时是否直接返回 false（默认 true）
                                 int attempts) const         // 最大重试次数（默认 10）
{
    CV_Assert(attempts > 0);  // 重试次数必须 > 0，否则无意义
    while (true)
    {
        errno = 0;
        int result = ioctl(deviceHandle, ioctlCode, parameter);  // 发起系统调用：fd + 命令码 + 参数
        int err = errno;                                          // 保存 errno，后续任何调用都可能覆盖它

        if (result != -1)     // ioctl 返回非 -1 = 成功
            return true;

        const bool isBusy = (err == EBUSY);  // EBUSY：另一个进程正在占用摄像头
        if (isBusy && failIfBusy)             // failIfBusy=true（默认）：EBUSY 直接放弃
            return false;

        if (!(isBusy || errno == EAGAIN))    // 非 EBUSY/EAGAIN 的错误（如 ENODEV 摄像头拔出）不可恢复
            return false;

        if (--attempts == 0)                  // 重试次数耗尽
            return false;

        // ── EAGAIN/EBUSY：帧还没准备好，调 select() 阻塞等待摄像头硬件就绪 ──
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(deviceHandle, &fds);           // 监听 /dev/video0 的可读事件（新帧到达）

        static int param_v4l_select_timeout =  // 超时 10 秒，可通过环境变量 OPENCV_VIDEOIO_V4L_SELECT_TIMEOUT 配置
            (int)utils::getConfigurationParameterSizeT("OPENCV_VIDEOIO_V4L_SELECT_TIMEOUT", 10);
        struct timeval tv;
        tv.tv_sec = param_v4l_select_timeout;  // 10 秒超时
        tv.tv_usec = 0;

        errno = 0;
        result = select(deviceHandle + 1, &fds, NULL, NULL, &tv);  // 阻塞等 fd 可读，底层系统调用是 pselect6
        err = errno;

        if (0 == result)        // select 返回 0 = 超时，10 秒内没有新帧
            return false;

        if (EINTR == err)       // 被信号中断（如 Ctrl+C），不继续重试
            return false;

        // select 返回 > 0：设备有数据了，回到 while 开头重新 ioctl
    }
    return true;
}
```

### ioctl 参数详解与完整调用链

**`ioctl(deviceHandle, ioctlCode, parameter)` 三个参数（以 VIDIOC_DQBUF 为例）：**

| 参数           | 值/类型               | 含义                                                 |
| -------------- | --------------------- | ---------------------------------------------------- |
| `deviceHandle` | `/dev/video0` 的 fd   | 文件描述符                                           |
| `ioctlCode`    | `0xC0565611`          | V4L2 出队命令（`'V'`=0x56, nr=17）                   |
| `parameter`    | `struct v4l2_buffer*` | 缓冲区信息（输入：index；输出：bytesused/timestamp） |

**cmd 编码方式（VIDIOC_DQBUF）：**
```c
// linux/include/uapi/linux/videodev2.h:2734
#define VIDIOC_DQBUF  _IOWR('V', 17, struct v4l2_buffer)

// 编码分解：
//   dir    = _IOC_READ | _IOC_WRITE = 3  → bits [31:30] = 11
//   type   = 'V' = 0x56                  → bits [29:16]
//   nr     = 17                          → bits [15:8]  = 0x11
//   size   = sizeof(v4l2_buffer)         → bits [7:0]
// cmd ≈ 0xC0565611
```

**struct v4l2_buffer 参数结构体：**
```c
struct v4l2_buffer {
    __u32     index;           // 输入：缓冲区编号
    __u32     type;            // 输入：V4L2_BUF_TYPE_VIDEO_CAPTURE
    __u32     bytesused;       // 输出：有效数据字节数（内核填充）
    __u32     flags;           // 输出：V4L2_BUF_FLAG_DONE
    __u32     memory;          // 输入：V4L2_MEMORY_MMAP
    union { __u32 offset; } m; // 输出：mmap 偏移量
    struct timeval timestamp;  // 输出：帧时间戳
    __u32     length;          // 缓冲区总长度
};
```

---

## 3. retrieveFrame: 从缓冲区到 NumPy 数组

```cpp
// opencv/modules/videoio/src/cap_v4l.cpp:2262
bool CvCaptureCAM_V4L::retrieveFrame(int, OutputArray ret)
{
    havePendingFrame = false;  // 重置多路同步旗标，lerobot 不用此标志

    if (bufferIndex < 0) {    // 无有效缓冲区（grab 失败后直接调 retrieve），返回上一帧
        frame.copyTo(ret);
        return true;
    }

    const Buffer &currentBuffer = buffers[bufferIndex];

    // lerobot convert_rgb=true（默认）：走 convertToRgb 做 MJPEG 解码 → BGR
    convertToRgb(currentBuffer);

    // 将缓冲区重新入队（VIDIOC_QBUF），供下次 grabFrame 使用
    // 这一步必须在 copyTo 之前，因为 grabFrame 只在 bufferIndex>=0 时才归还旧缓冲区
    if (!tryIoctl(VIDIOC_QBUF, &buffers[bufferIndex].buffer)) {
        CV_LOG_DEBUG(NULL, "VIDEOIO(V4L2:" << deviceName
            << "): failed VIDIOC_QBUF: errno=" << errno);
    }

    bufferIndex = -1;              // 归还完毕，标记无持有缓冲区
    frame.copyTo(ret);             // frame（cv::Mat BGR）→ OutputArray ret → Python 侧 np.ndarray
    return true;
}
```

### frame.copyTo → np.ndarray 转换细节

`frame.copyTo(ret)` 这里的 `ret` 是 `OutputArray`，Python 调用侧它对应一个 numpy 数组。转换路径在 `modules/python/src2/cv2_convert.cpp` 和 `cv2_numpy.cpp` 里。

#### 第一步：NumpyAllocator::allocate 分配 ndarray 内存

`copyTo` 内部调 `mat.create()`，`create` 发现需要分配新内存，调 `allocator->allocate()`。Python 绑定模块加载时已把全局 allocator 换成 `NumpyAllocator`，所以走到这里：

```cpp
// opencv/modules/python/src2/cv2_numpy.cpp:24
// 参数：dims0=2（H,W 两维，channel 单独处理），sizes=[360,640]，type=CV_8UC3
UMatData* NumpyAllocator::allocate(int dims0, const int* sizes, int type,
                                   void* data, size_t* step,
                                   AccessFlag flags, UMatUsageFlags usageFlags) const
{
    // data==0 说明需要新建 numpy 数组（从 copyTo 路径进来时始终为 0）
    PyEnsureGIL gil;

    int depth = CV_MAT_DEPTH(type);   // CV_8UC3 → depth = CV_8U = 0
    int cn    = CV_MAT_CN(type);      // CV_8UC3 → cn = 3

    // CV_8U → NPY_UBYTE；其他深度类似映射
    int typenum = depth == CV_8U ? NPY_UBYTE : /* ... */;

    int dims = dims0;                          // dims=2
    cv::AutoBuffer<npy_intp> _sizes(dims + 1);
    for (int i = 0; i < dims; i++)
        _sizes[i] = sizes[i];                  // _sizes = [360, 640]
    if (cn > 1)
        _sizes[dims++] = cn;                   // cn=3>1 → _sizes = [360,640,3]，dims=3

    // 分配 Python 堆上的 numpy 数组：shape=(360,640,3)，dtype=uint8
    // 内部 strides 自动算为 [1920, 3, 1]（C-contiguous）
    PyObject* o = PyArray_SimpleNew(dims, _sizes.data(), typenum);

    // 复用同文件中的另一个 allocate 重载，把 PyObject* 挂进 UMatData
    return allocate(o, dims0, sizes, type, step);
    //              ↑ 见下面第一个重载，设置 u->data/origdata/userdata/size/step
}

// opencv/modules/python/src2/cv2_numpy.cpp:11
// 把已有 PyObject* 包装成 UMatData（Mat 的内存管理块）
UMatData* NumpyAllocator::allocate(PyObject* o, int dims,
                                   const int* sizes, int type, size_t* step) const
{
    UMatData* u  = new UMatData(this);
    u->data      = u->origdata = (uchar*)PyArray_DATA((PyArrayObject*)o);
    //              ↑ Mat.data 直接指向 ndarray 的像素内存区，两者共享同一块物理内存

    npy_intp* _strides = PyArray_STRIDES((PyArrayObject*)o);
    for (int i = 0; i < dims - 1; i++)
        step[i] = (size_t)_strides[i];        // step[0]=1920（行步长），step[1]=3（列步长）
    step[dims-1] = CV_ELEM_SIZE(type);         // 最内层步长 = 单像素字节数 = 3

    u->size     = sizes[0] * step[0];          // 360 × 1920 = 691200 字节
    u->userdata = o;                           // 把 PyObject*(ndarray) 挂在 userdata，供后续取出
    return u;
}
```

此时 `frame.data` 指向 ndarray 的像素内存，`copyTo` 把 libjpeg 解码好的 BGR 像素 **memcpy 691,200 字节**到这块内存（整个管线唯一的一次像素拷贝）。

#### 第二步：NumpyAllocator::deallocate 管理生命周期

```cpp
// opencv/modules/python/src2/cv2_numpy.cpp:58
void NumpyAllocator::deallocate(UMatData* u) const
{
    if (!u) return;
    PyEnsureGIL gil;
    CV_Assert(u->urefcount >= 0);
    CV_Assert(u->refcount  >= 0);
    if (u->refcount == 0)
    {
        PyObject* o = (PyObject*)u->userdata;
        Py_XDECREF(o);   // C++ 侧 frame 析构时减引用计数；归零后 numpy GC 才真正 free
        delete u;
    }
}
```

#### 第三步：pyopencv_from 把 Mat 变成 Python 对象

Python 调用 `cap.read()` 拿到返回值时，OpenCV Python 绑定调这个模板函数：

```cpp
// opencv/modules/python/src2/cv2_convert.cpp:322
template<>
PyObject* pyopencv_from(const cv::Mat& m)
{
    if (!m.data)
        Py_RETURN_NONE;

    cv::Mat temp, *p = (cv::Mat*)&m;

    // 检查 allocator：如果不是 NumpyAllocator，需要先拷一次到新的 numpy-backed Mat
    if (!p->u || p->allocator != &GetNumpyAllocator())
    {
        temp.allocator = &GetNumpyAllocator();
        ERRWRAP2(m.copyTo(temp));   // 只在 allocator 不匹配时才触发额外拷贝
        p = &temp;
    }
    // frame 经过上面 copyTo 已是 NumpyAllocator，allocator 匹配，直接跳过

    PyObject* o = (PyObject*)p->u->userdata;   // 取出之前挂的 ndarray PyObject*
    Py_INCREF(o);   // 引用计数 +1，Python 侧持有这个对象，防止 C++ 析构时提前 free
    return o;       // 返回给 Python 解释器，即 ret, frame = cap.read() 中的 frame
}
```

```python
ret, frame = self.videocapture.read()   # 阻塞，等待 V4L2 下一帧（~33ms@30fps）
```

#### 转换结果总结

Python 侧拿到的 `np.ndarray`：

| 属性      | 值                                                |
| --------- | ------------------------------------------------- |
| `shape`   | `(360, 640, 3)`                                   |
| `dtype`   | `uint8`                                           |
| `strides` | `(1920, 3, 1)`（行/列/通道步长，字节）            |
| 通道顺序  | BGR（`[...,0]`=B，`[...,1]`=G，`[...,2]`=R）      |
| 内存      | Python 堆，numpy GC 管理；C++ 析构时 `Py_XDECREF` |

**关键点**：整个流程**只有一次 691,200 字节的像素拷贝**（libjpeg 逐行写进 `frame` 内部缓冲 → `copyTo` memcpy 到 numpy 内存）。mmap→libjpeg→frame 是零拷贝（直接写），frame→ndarray 是一次 memcpy，ndarray→Python 是零拷贝（同一块内存的 PyObject* 传递）。

---

## 4. convertToRgb: 格式转换详解（lerobot MJPEG 路径）

```cpp
// opencv/modules/videoio/src/cap_v4l.cpp:1553
void CvCaptureCAM_V4L::convertToRgb(const Buffer &currentBuffer)
{
    cv::Size imageSize;
    unsigned char *start;

    // ── 确定数据起始地址（lerobot 单平面 MJPEG）──
    start = (unsigned char*)currentBuffer.memories[MEMORY_ORIG].start;  // 直接指向 mmap 地址，零拷贝
    imageSize = cv::Size(form.fmt.pix.width, form.fmt.pix.height);     // 配置的分辨率，如 640×360

    frame.create(imageSize, CV_8UC3);  // 分配/复用输出 BGR 帧，shape = (H, W, 3)

    // ── MJPEG 解码（lerobot 每帧走这里）──
    // imdecode 内部调用 libjpeg：Huffman 解码 + IDCT + 色彩空间转换 → BGR
    // ARM CPU 上这是整个管线最重的操作（10-30ms/帧@1080p，640×360 约 3-8ms）
    cv::imdecode(Mat(1, currentBuffer.bytesused, CV_8U, start),  // 输入：MJPEG 压缩数据，1×bytesused 行，之前填充元数据的时候内核填充的
                 IMREAD_COLOR,                                     // 输出要求：BGR 三通道
                 &frame);                                          // 输出目标：frame（cv::Mat BGR）
}
```

**`cv::imdecode` 三个参数详解（lerobot 实际值）：**

| 参数    | lerobot 实际值                                                                                         | 类型         | 含义                                                                                                                                                  |
| ------- | ------------------------------------------------------------------------------------------------------ | ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `_buf`  | `Mat(1, bytesused, CV_8U, start)`，bytesused ≈ 20-60KB（640×360 MJPEG 每帧压缩大小，随画面复杂度浮动） | `InputArray` | 把 mmap 里的压缩数据**零拷贝包装**成 1×N 的 `cv::Mat`：`1`=1行，`bytesused`=列数即压缩字节数，`CV_8U`=元素类型 uint8，`start`=mmap 指针直接用，不复制 |
| `flags` | `IMREAD_COLOR`（值=1）                                                                                 | `int`        | 解码目标格式：BGR 三通道；libjpeg 输出后 lerobot 的 `_postprocess_image` 再做 BGR→RGB 转换                                                            |
| `dst`   | `&frame`                                                                                               | `Mat*`       | 输出目标：解码结果直接写入 `frame`（`cv::Mat`，shape=**360×640×3**，dtype=uint8，BGR）；解码完成后 `frame.copyTo(ret)` 转成 Python np.ndarray         |

---

**`cv::Mat` 数据结构定义及 lerobot 中两个实例的实际值：**

Mat 是一个"胖指针"：轻量 header + 裸像素指针，header 在栈上，像素在堆/mmap 上。

```cpp
// opencv/modules/core/include/opencv2/core/mat.hpp:2200
int flags;      // 魔数 | 连续标志 | type 编码
                // type = depth(低3位) + (cn-1)<<3
                // _buf:  CV_8U   = 0
                // frame: CV_8UC3 = 0 + (3-1)<<3 = 16

int dims;       // 维度数，通常 = 2
                // _buf: 2 / frame: 2

int rows, cols; // 行 × 列
                // _buf:  rows=1,        cols=bytesused（≈20-60K，MJPEG 压缩字节数，随画面浮动）
                // frame: rows=360,      cols=640

uchar* data;    // 裸像素指针，不管理生命周期
                // _buf:  start（mmap 地址，直接指向内核 DMA 缓冲，零拷贝）
                // frame: 堆分配的 BGR 像素块起点（copyTo 后指向 numpy 内存）

const uchar* datastart;   // ROI 边界，submatrix/crop 时用；非 ROI 时三者覆盖同一区域
const uchar* dataend;
const uchar* datalimit;

MatAllocator* allocator;  // 内存分配策略
                          // _buf:  nullptr（用默认堆分配器，但实际上 data 不由它分配）
                          // frame: 初始 nullptr，copyTo 后换成 NumpyAllocator（Python 堆）

UMatData* u;    // 引用计数块（见下方 UMatData），多 Mat 共享同一像素块时用
                // _buf:  nullptr（不持有内存，析构不 free，只是 mmap 的视图）
                // frame: UMatData*，copyTo 后 u->userdata = PyObject*（ndarray 对象）

MatSize size;   // size[i] 即 sizes 数组（size[0]=rows, size[1]=cols）
MatStep step;   // step[0]=每行字节数, step[1]=单元素字节数
                // _buf:  step[0]=bytesused, step[1]=1
                // frame: step[0]=1920（=640×3）, step[1]=3
```

`UMatData` 引用计数块（`mat.hpp:596`），`frame` 的 `u` 指向一个这样的结构：

```cpp
// opencv/modules/core/include/opencv2/core/mat.hpp:596
const MatAllocator* prevAllocator;
const MatAllocator* currAllocator;
int urefcount;    // UMat 引用计数（GPU/异构内存用）
int refcount;     // CPU 侧引用计数，归零时 deallocate 被调用
uchar* data;      // 同 Mat::data，指向像素内存起点（= PyArray_DATA(o)）
uchar* origdata;  // 原始分配起点（同 data，无偏移时相同）
size_t size;      // 总字节数 = 360×1920 = 691,200
UMatData::MemoryFlag flags;
void* handle;
void* userdata;   // NumpyAllocator 把 PyObject*(ndarray) 挂在这里，pyopencv_from 取出它
int allocatorFlags_;
int mapcount;
UMatData* originalUMatData;
```

---

**调用链：`cv::imdecode` → `imdecode_` → `JpegDecoder`**

```cpp
// opencv/modules/imgcodecs/src/loadsave.cpp:1455
// 这是 cv::imdecode(buf, flags, &frame) 实际进入的重载（带 dst 指针）
Mat imdecode( InputArray _buf,   // 压缩数据（包装了 mmap 的 cv::Mat）
              int flags,          // IMREAD_COLOR = 1，要求 BGR 输出
              Mat* dst )          // &frame，解码结果写这里
{
    Mat buf = _buf.getMat(), img;
    dst = dst ? dst : &img;                          // dst 非空，直接用 &frame
    if (imdecode_(buf, flags, *dst, nullptr, noArray()))  // 调内部实现
        return *dst;
    else
        return cv::Mat();
}

// opencv/modules/imgcodecs/src/loadsave.cpp:1306
// 真正的解码逻辑
static bool imdecode_( const Mat& buf,   // 1×bytesused 的 CV_8U Mat（MJPEG 压缩数据）
                        int flags,        // IMREAD_COLOR
                        Mat& mat,         // frame，解码结果写入目标
                        ... )
{
    // ① 嗅探文件头，从已注册的 decoder 列表里找匹配的解码器
    // findDecoder 读取 buf 头部几个字节，比对 JPEG 魔数（0xFF 0xD8），返回 JpegDecoder 实例
    ImageDecoder decoder = findDecoder(buf_row);

    // ② 把 buf 的内存地址直接告诉解码器（setSource），不拷贝数据
    decoder->setSource(buf_row);

    // ③ 读头部：调 JpegDecoder::readHeader()
    //    内部调 libjpeg C API：jpeg_create_decompress + jpeg_read_header
    //    读取图像宽高、颜色空间等元数据，写入 decoder->m_width/m_height/m_type
    decoder->readHeader();

    // 根据 flags 和 decoder 报告的原始类型，计算输出 Mat 的 type（CV_8UC3）
    const int type = calcType(decoder->type(), flags);
    mat.create( size.height, size.width, type );  // frame 在这里分配/复用内存

    // ④ 读像素：调 JpegDecoder::readData()
    //    内部调 libjpeg：jpeg_start_decompress + 逐行 jpeg_read_scanlines
    //    每行像素直接写到 mat.ptr<uchar>(iy)，无中间缓冲
    decoder->readData(mat);

    return true;
}

// opencv/modules/imgcodecs/src/grfmt_jpeg.cpp:437
// JpegDecoder::readData：libjpeg 单核 CPU 软件解码
// ─────────────────────────────────────────────────────────────────────
// 执行模型：单线程，跑在调用 imdecode 的那个 CPU 核心上，全程阻塞直到解码完毕。
// 树莓派5 有 VideoCore VII GPU 硬件视频解码器，但它只支持 H.264/H.265，
// MJPEG 不在硬件路径上，因此 imdecode 走不到 GPU，只能软解。
// libjpeg-turbo 在 ARM 上用 NEON SIMD（simd/arm/ 目录下的汇编），
// IDCT 和 YCbCr→BGR 色彩转换有向量加速，但仍是 CPU 单核运算，
// 640×360 每帧约 3-8ms，1080p 约 10-30ms。
// ─────────────────────────────────────────────────────────────────────
bool JpegDecoder::readData( Mat& img )
{
    jpeg_decompress_struct* cinfo = &((JpegState*)m_state)->cinfo;

    // MJPEG 可能缺标准 Huffman 表（某些摄像头省略），手动补全
    // （libjpeg-turbo >= 1.3.90 版本自己处理，旧版需要 CV_MANUAL_JPEG_STD_HUFF_TABLES）
    if ( cinfo->ac_huff_tbl_ptrs[0] == NULL && ... )
        my_jpeg_load_dht( cinfo, my_jpeg_odml_dht, ... );

    // 设置输出色彩空间：IMREAD_COLOR → JCS_EXT_BGR（libjpeg-turbo）或 JCS_RGB（原版）
    // libjpeg-turbo 支持 JCS_EXT_BGR，可直接输出 BGR，省去一次 RGB→BGR 转换
    cinfo->out_color_space = JCS_EXT_BGR;

    jpeg_start_decompress( cinfo );   // 启动解码引擎（IDCT 等初始化）

    // 单核 CPU 单线程逐行解码，360 行全部跑完才返回（640×360 约 3-8ms）
    // 每行：Huffman 解码 → IDCT（NEON 加速）→ YCbCr→BGR（NEON 加速）→ 写入 img
    for( int iy = 0; iy < m_height; iy++ )
    {
        uchar* data = img.ptr<uchar>(iy);
        // jpeg_read_scanlines：解码一行 MJPEG → BGR 像素，写入 data
        // 参数：cinfo（解码器状态），&data（行指针数组），1（本次解码行数）
        jpeg_read_scanlines( cinfo, &data, 1 );
    }
}
```

> lerobot 强制 `FOURCC=MJPG`，`convertToRgb` 只走 MJPEG 分支。源码中还有 YVU420/YUYV/NV12 等格式的 `cvtColor` 分支，lerobot 不会触发。

---

## 5. 推理路径总结

```
videocapture.read()  (每帧调用一次)
    │
    ├─ grab()
    │    └─ grabFrame()
    │         └─ tryIoctl(VIDIOC_DQBUF)
    │              ├─ ioctl(VIDIOC_DQBUF)  → 成功则返回
    │              └─ EAGAIN/EBUSY → select/pselect6 等待 (~33ms@30fps)
    │
    └─ retrieve()
         └─ retrieveFrame()
              ├─ convertToRgb()
              │    └─ MJPEG: imdecode()      ← 3-8ms@640×360，最大瓶颈
              │
              └─ ioctl(VIDIOC_QBUF)          ← 归还缓冲区
```

**推理时的系统调用只有 2 个 ioctl**：
1. `VIDIOC_DQBUF` — 取已填充的缓冲区
2. `VIDIOC_QBUF` — 归还缓冲区

**最耗时的部分**：MJPEG 解码（libjpeg，10-30ms@1080p），纯 CPU 计算。
