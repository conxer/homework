import cv2
import numpy as np
import gradio as gr

# 初始化全局变量，存储控制点和目标点
points_src = []
points_dst = []
image = None

# 上传图像时清空控制点和目标点
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()  # 清空控制点
    points_dst.clear()  # 清空目标点
    image = img
    return img

# 记录点击点事件，并标记点在图像上，同时在成对的点间画箭头
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]  # 获取点击的坐标
    
    # 判断奇偶次来分别记录控制点和目标点
    if len(points_src) == len(points_dst):
        points_src.append([x, y])  # 奇数次点击为控制点
    else:
        points_dst.append([x, y])  # 偶数次点击为目标点
    
    # 在图像上标记点（蓝色：控制点，红色：目标点），并画箭头
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # 蓝色表示控制点
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # 红色表示目标点
    
    # 画出箭头，表示从控制点到目标点的映射
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)  # 绿色箭头表示映射
    
    return marked_image

# 执行仿射变换
from scipy.spatial.distance import cdist


def point_guided_deformation(im, psrc, pdst):
    """
    使用径向基函数RBF进行图像变形并进行插值处理填补空白。

    参数:
    - im: 输入图像NumPy 数组格式。
    - psrc: 源控制点 (N x 2) 数组。
    - pdst: 目标控制点 (N x 2) 数组。

    返回:
    - im2: 变形后的输出图像。
    """
    # RBF 参数，用于控制变形的平滑度
    d = 500

    # 计算源点的距离矩阵并构建线性系统矩阵
    A = 1.0 / (cdist(psrc, psrc, 'sqeuclidean') + d)

    # 求解变形系数
    coef = np.linalg.solve(A, pdst - psrc)

    # 获取图像的尺寸（高，宽）
    h, w = im.shape[:2]
    xpix, ypix = np.meshgrid(np.arange(1, w + 1), np.arange(1, h + 1))
    x = np.vstack([xpix.ravel(), ypix.ravel()]).T

    # 为每个源像素计算变形后的坐标
    B = 1.0 / (cdist(x, psrc, 'sqeuclidean') + d)
    q0 = np.dot(B, coef) + x

    # 将变形后的坐标值限制在图像范围内
    q = np.clip(np.ceil(q0).astype(int), 1, [w, h])

    # 创建一个标志，用于过滤在图像范围内的有效像素
    pflag = np.all(q > 0, axis=1) & (q[:, 0] <= w) & (q[:, 1] <= h)

    # 创建从目标像素到源像素的映射
    mapPId = np.ravel_multi_index((q[pflag, 1] - 1, q[pflag, 0] - 1), (h, w))

    # 将输入图像展平，并准备一个空的结果图像
    im = im.reshape(h * w, 3)
    im2 = np.ones((h * w, 3), dtype=np.uint8) * 255  # 初始化为白色背景

    # 映射有效像素
    im2[mapPId, :] = im[pflag, :]
    im2 = im2.reshape(h, w, 3)

    # ------ 插值处理，填补空白区域 ------
    # 首先，将图像转换为灰度图并生成一个掩码，掩码为白色背景（255）的区域
    gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    mask = (gray == 255).astype(np.uint8)  # 掩码，白色部分

    # 使用 `inpaint` 方法进行插值修复
    im2_filled = cv2.inpaint(im2, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    return im2_filled




def run_warping():
    global points_src, points_dst, image ### 获取全局变量
    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# 清除选中点
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image  # 返回未标记的原图

# 使用 Gradio 构建界面
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source="upload", label="上传图片", interactive=True, width=800, height=200)
            point_select = gr.Image(label="点击选择控制点和目标点", interactive=True, width=800, height=800)
            
        with gr.Column():
            result_image = gr.Image(label="变换结果", width=800, height=400)
    
    # 按钮
    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")  # 添加清除按钮
    
    # 上传图像的交互
    input_image.upload(upload_image, input_image, point_select)
    # 选择点的交互，点选后刷新图像
    point_select.select(record_points, None, point_select)
    # 点击运行 warping 按钮，计算并显示变换后的图像
    run_button.click(run_warping, None, result_image)
    # 点击清除按钮，清空所有已选择的点
    clear_button.click(clear_points, None, point_select)
    
# 启动 Gradio 应用
demo.launch()