import gradio as gr
import cv2
import numpy as np

# Function to convert 2x3 affine matrix to 3x3 for matrix multiplication
def to_3x3(affine_matrix):
    return np.vstack([affine_matrix, [0, 0, 1]])

# Function to apply transformations based on user inputs
def apply_transform(image, scale, rotation, translation_x, translation_y, flip_horizontal):
    # 将图像从 PIL 格式转换为 NumPy 数组
    image = np.array(image)

    # 获取原图像的中心
    original_center = (image.shape[1] // 2, image.shape[0] // 2)

    # 计算扩展后的图像尺寸
    new_width = int(image.shape[1] * 2)
    new_height = int(image.shape[0] * 2)
    
    # 创建一个新的图像，填充为白色
    image_new = np.ones((new_height, new_width, 3), dtype=np.uint8) * 255

    # 将原图像放置在新图像的中心
    image_new[original_center[1]:original_center[1] + image.shape[0],
              original_center[0]:original_center[0] + image.shape[1]] = image

    # 获取新的中心位置
    new_center = (new_width // 2, new_height // 2)

    # 创建缩放矩阵
    M_scale = cv2.getRotationMatrix2D(new_center, 0, scale)  # 缩放矩阵
    # 创建旋转矩阵
    M_rotate = cv2.getRotationMatrix2D(new_center, rotation, 1)  # 旋转矩阵

    # 合并缩放和旋转矩阵
    M_combined = np.vstack([M_rotate, [0, 0, 1]]) @ to_3x3(M_scale)

    # 应用平移
    M_combined[0, 2] += translation_x
    M_combined[1, 2] += translation_y

    # 检查是否需要水平翻转
    if flip_horizontal:
        # 创建翻转矩阵
        M_flip = np.array([[-1, 0, new_width], [0, 1, 0], [0, 0, 1]])
        M_combined = M_combined @ M_flip  # 合并翻转矩阵

    # 执行变换
    transformed_image = cv2.warpAffine(image_new, M_combined[:2], (new_width, new_height), borderValue=(255, 255, 255))

    return transformed_image



# Gradio Interface
def interactive_transform():
    with gr.Blocks() as demo:
        gr.Markdown("## Image Transformation Playground")
        
        # Define the layout
        with gr.Row():
            # Left: Image input and sliders
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Image")

                scale = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, value=1.0, label="Scale")
                rotation = gr.Slider(minimum=-180, maximum=180, step=1, value=0, label="Rotation (degrees)")
                translation_x = gr.Slider(minimum=-300, maximum=300, step=10, value=0, label="Translation X")
                translation_y = gr.Slider(minimum=-300, maximum=300, step=10, value=0, label="Translation Y")
                flip_horizontal = gr.Checkbox(label="Flip Horizontal")
            
            # Right: Output image
            image_output = gr.Image(label="Transformed Image")
        
        # Automatically update the output when any slider or checkbox is changed
        inputs = [
            image_input, scale, rotation, 
            translation_x, translation_y, 
            flip_horizontal
        ]

        # Link inputs to the transformation function
        image_input.change(apply_transform, inputs, image_output)
        scale.change(apply_transform, inputs, image_output)
        rotation.change(apply_transform, inputs, image_output)
        translation_x.change(apply_transform, inputs, image_output)
        translation_y.change(apply_transform, inputs, image_output)
        flip_horizontal.change(apply_transform, inputs, image_output)

    return demo

# Launch the Gradio interface
interactive_transform().launch()
