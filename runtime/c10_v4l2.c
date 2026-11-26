#define _POSIX_C_SOURCE 200809L

#include "c10_v4l2.h"

#include <errno.h>
#include <fcntl.h>
#include <time.h>
#include <linux/videodev2.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

enum { BUFFER_COUNT = 4, MAX_BUFFER_COUNT = 8 };
struct c10_camera {
    int fd, streaming;
    unsigned width, height, count;
    size_t stride;
    c10_pixfmt format;
    struct { void *ptr; size_t bytes; } buffers[MAX_BUFFER_COUNT];
};

static int camera_ioctl(int fd, unsigned long request, void *arg)
{
    int result;
    do {
        result = ioctl(fd, request, arg);
    } while (result < 0 && errno == EINTR);
    return result;
}

static int queue_buffer(c10_camera *cam, unsigned index)
{
    struct v4l2_buffer buf = { .type = V4L2_BUF_TYPE_VIDEO_CAPTURE,
                               .memory = V4L2_MEMORY_MMAP, .index = index };
    return camera_ioctl(cam->fd, VIDIOC_QBUF, &buf);
}

int c10_camera_payload_valid(unsigned flags, size_t used, size_t mapped)
{
    return !(flags & V4L2_BUF_FLAG_ERROR) && used && used <= mapped;
}

static int select_format(c10_camera *cam, unsigned w, unsigned h)
{
    const uint32_t order[] = { V4L2_PIX_FMT_YUYV, V4L2_PIX_FMT_RGB24, V4L2_PIX_FMT_MJPEG };
    for (size_t i = 0; i < sizeof(order) / sizeof(order[0]); ++i) {
        struct v4l2_format fmt = { .type = V4L2_BUF_TYPE_VIDEO_CAPTURE };
        fmt.fmt.pix.width = w;
        fmt.fmt.pix.height = h;
        fmt.fmt.pix.field = V4L2_FIELD_ANY;
        fmt.fmt.pix.pixelformat = order[i];
        if (camera_ioctl(cam->fd, VIDIOC_S_FMT, &fmt) || fmt.fmt.pix.pixelformat != order[i])
            continue;
        struct v4l2_pix_format *pix = &fmt.fmt.pix;
        size_t ignored;
        if (c10_frame_size(pix->width, pix->height, &ignored) ||
            (i == C10_YUYV && (pix->width & 1)))
            continue;
        cam->width = pix->width;
        cam->height = pix->height;
        cam->stride = pix->bytesperline;
        cam->format = (c10_pixfmt)i;
        fprintf(stderr, "V4L2 %s %ux%u, requested %ux%u\n",
                i == C10_YUYV ? "YUYV" : i == C10_RGB24 ? "RGB24" : "baseline MJPEG",
                cam->width, cam->height, w, h);
        return 0;
    }
    fprintf(stderr, "V4L2 device did not negotiate YUYV, RGB24 or baseline MJPEG within %ux%u\n",
            C10_MAX_WIDTH, C10_MAX_HEIGHT);
    errno = ENOTSUP;
    return -1;
}

c10_camera *c10_camera_open(const char *device, unsigned w, unsigned h, unsigned fps)
{
    c10_camera *cam = calloc(1, sizeof(*cam));
    struct v4l2_capability caps = {0};
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (!cam)
        return NULL;
    cam->fd = open(device, O_RDWR | O_NONBLOCK | O_CLOEXEC);
    if (cam->fd < 0 || camera_ioctl(cam->fd, VIDIOC_QUERYCAP, &caps))
        goto fail;
    uint32_t features = caps.capabilities & V4L2_CAP_DEVICE_CAPS ? caps.device_caps : caps.capabilities;
    uint32_t needed = V4L2_CAP_VIDEO_CAPTURE | V4L2_CAP_STREAMING;
    if ((features & needed) != needed) {
        errno = ENOTSUP;
        goto fail;
    }
    if (select_format(cam, w, h))
        goto fail;
    struct v4l2_streamparm parm = { .type = V4L2_BUF_TYPE_VIDEO_CAPTURE };
    parm.parm.capture.timeperframe.numerator = 1;
    parm.parm.capture.timeperframe.denominator = fps;
    if (camera_ioctl(cam->fd, VIDIOC_S_PARM, &parm))
        fprintf(stderr, "V4L2 kept the driver frame interval (requested %u fps)\n", fps);
    else if (parm.parm.capture.timeperframe.numerator)
        fprintf(stderr, "V4L2 frame interval %u/%u s\n", parm.parm.capture.timeperframe.numerator,
                parm.parm.capture.timeperframe.denominator);

    struct v4l2_requestbuffers request = { .count = BUFFER_COUNT,
        .type = V4L2_BUF_TYPE_VIDEO_CAPTURE, .memory = V4L2_MEMORY_MMAP };
    if (camera_ioctl(cam->fd, VIDIOC_REQBUFS, &request))
        goto fail;
    if (request.count < 2 || request.count > MAX_BUFFER_COUNT) {
        errno = ENOBUFS;
        goto fail;
    }
    for (unsigned i = 0; i < request.count; ++i) {
        struct v4l2_buffer buf = { .type = V4L2_BUF_TYPE_VIDEO_CAPTURE,
                                   .memory = V4L2_MEMORY_MMAP, .index = i };
        if (camera_ioctl(cam->fd, VIDIOC_QUERYBUF, &buf))
            goto fail;
        if (!buf.length || buf.length > 2u * C10_MAX_RGB) {
            errno = EMSGSIZE;
            goto fail;
        }
        void *ptr = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, cam->fd, buf.m.offset);
        if (ptr == MAP_FAILED)
            goto fail;
        cam->buffers[i].ptr = ptr;
        cam->buffers[i].bytes = buf.length;
        cam->count++;
        if (queue_buffer(cam, buf.index))
            goto fail;
    }
    if (camera_ioctl(cam->fd, VIDIOC_STREAMON, &type))
        goto fail;
    cam->streaming = 1;
    return cam;

fail:
    {
        int saved = errno;
        c10_camera_close(cam);
        errno = saved;
        return NULL;
    }
}

/* Returns 1 for a frame, 0 for a poll interruption or a recoverable dropped frame. */
int c10_camera_rgb(c10_camera *cam, uint8_t *rgb, size_t capacity, unsigned *w, unsigned *h)
{
    struct pollfd pfd = { .fd = cam->fd, .events = POLLIN };
    int ready = poll(&pfd, 1, 2000);
    if (ready < 0 && errno == EINTR)
        return 0;
    if (ready <= 0) {
        if (ready == 0)
            errno = ETIMEDOUT;
        return -1;
    }
    if (!(pfd.revents & POLLIN)) {
        errno = EIO;
        return -1;
    }
    struct v4l2_buffer buf = { .type = V4L2_BUF_TYPE_VIDEO_CAPTURE, .memory = V4L2_MEMORY_MMAP };
    if (camera_ioctl(cam->fd, VIDIOC_DQBUF, &buf))
        return errno == EAGAIN ? 0 : -1;
    /* A 30 fps camera routinely outruns inference. Drain completed buffers
       and convert the newest available image, with one buffer held until copy. */
    for (unsigned i = 1; i < cam->count; ++i) {
        struct v4l2_buffer newer = { .type = V4L2_BUF_TYPE_VIDEO_CAPTURE,
                                     .memory = V4L2_MEMORY_MMAP };
        if (camera_ioctl(cam->fd, VIDIOC_DQBUF, &newer)) {
            if (errno == EAGAIN)
                break;
            return -1;
        }
        if (buf.index >= cam->count) {
            errno = EIO;
            return -1;
        }
        if (queue_buffer(cam, buf.index))
            return -1;
        buf = newer;
    }
    if (buf.index >= cam->count) {
        errno = EIO;
        return -1;
    }
    int bad = !c10_camera_payload_valid(buf.flags, buf.bytesused, cam->buffers[buf.index].bytes);
    if (!bad)
        bad = c10_pixels_rgb(cam->format, cam->width, cam->height, cam->stride,
                             cam->buffers[buf.index].ptr, buf.bytesused, rgb, capacity);
    if (queue_buffer(cam, buf.index))
        return -1;
    if (bad) {
        fprintf(stderr, "V4L2 dropped invalid %s frame (%u bytes)\n",
                cam->format == C10_JPEG ? "baseline MJPEG" : "uncompressed", buf.bytesused);
        return 0;
    }
    *w = cam->width;
    *h = cam->height;
    return 1;
}

void c10_camera_close(c10_camera *cam)
{
    if (!cam)
        return;
    if (cam->streaming) {
        enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        camera_ioctl(cam->fd, VIDIOC_STREAMOFF, &type);
    }
    for (unsigned i = 0; i < cam->count; ++i)
        munmap(cam->buffers[i].ptr, cam->buffers[i].bytes);
    if (cam->fd >= 0)
        close(cam->fd);
    free(cam);
}
