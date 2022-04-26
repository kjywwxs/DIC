import tkinter as tk
from tkinter import ttk
import DIC
import matplotlib.pyplot as plt
from matplotlib.image import imread
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

root = tk.Tk()
root.title('DIC位移计算 V0.0.1')
root.geometry('1000x600+100+100')
'''
左边布局
'''
left_frame = tk.Frame(root)
left_frame.pack(side=tk.LEFT, anchor=tk.N, padx=5, pady=5)

'''左边布局,参数设置'''
params_frame = tk.LabelFrame(left_frame, text='参数设置', padx=5, pady=5)
params_frame.pack()
tk.Label(params_frame, text='(1)方形子区大小').pack(anchor=tk.W)
entry_sub_size = tk.Entry(params_frame).pack(fill=tk.X)

tk.Label(params_frame, text='(2)计算点步长').pack(anchor=tk.W)
entry_step = tk.Entry(params_frame).pack(fill=tk.X)

tk.Label(params_frame, text='(3)是否归一化').pack(anchor=tk.W)
Normalization = ttk.Combobox(params_frame)
Normalization['values'] = ['是', '否']
Normalization.current(0)
Normalization.pack()

# tk.Label(params_frame, text='(4)整像素方法').pack(anchor=tk.W)
# zheng_fun = ttk.Combobox(params_frame)
# zheng_fun['values'] = ['traverse', 'GA', '十字搜索', '手动给定']
# zheng_fun.current(3)
# zheng_fun.pack()
#
# tk.Label(params_frame, text='(4)亚像素方法').pack(anchor=tk.W)
# ya_fun = ttk.Combobox(params_frame)
# ya_fun['values'] = ['IC-GN', 'IC-GN2']
# ya_fun.current(0)
# ya_fun.pack()

'''左边布局,整像素方法'''
zheng_method = tk.StringVar()
zheng_frame = tk.LabelFrame(left_frame, text='初始点整像素方法', padx=5, pady=5)
zheng_frame.pack(fill=tk.X)
tk.Radiobutton(zheng_frame, text='traverse', variable=zheng_method, value='traverse').pack(anchor=tk.W)
tk.Radiobutton(zheng_frame, text='GA', variable=zheng_method, value='GA').pack(anchor=tk.W)
tk.Radiobutton(zheng_frame, text='十字搜索', variable=zheng_method, value='十字搜索').pack(anchor=tk.W)
tk.Radiobutton(zheng_frame, text='手动给定', variable=zheng_method, value='手动给定').pack(anchor=tk.W)

'''左边布局,亚像素方法'''
ya_method = tk.StringVar()
ya_frame = tk.LabelFrame(left_frame, text='亚像素方法', padx=5, pady=5)
ya_frame.pack(fill=tk.X)
tk.Radiobutton(ya_frame, text='IC-GN', variable=ya_method, value='IC-GN').pack(anchor=tk.W)
tk.Radiobutton(ya_frame, text='IC-GN2', variable=ya_method, value='IC-GN2').pack(anchor=tk.W)

'''左边布局,打开文件按钮'''
button_frame = tk.Frame(left_frame, padx=5, pady=5)
button_frame.pack()
open_ref_img = tk.Button(button_frame, text='打开参考图像').pack(side=tk.LEFT)
open_tar_img = tk.Button(button_frame, text='打开目标图像').pack(side=tk.RIGHT)
'''左边布局,计算按钮'''
calculate = tk.Button(left_frame, text='计算').pack(fill=tk.X, padx=5, pady=5)

'''
右边布局，上面参考和目标图像，下面目标图像xy位移
'''
ref_img_dict = ['data/x y方向位移虚拟散斑/15.23pix-y方向位移/ImgSpeck1.bmp',
                'data/x y方向位移虚拟散斑/18.23pix-y方向 15.8pix-x方向/ImgSpeck1.bmp']
tar_img_dict = ['data/x y方向位移虚拟散斑/15.23pix-y方向位移/ImgSpeck2.bmp',
                'data/x y方向位移虚拟散斑/18.23pix-y方向 15.8pix-x方向/ImgSpeck2.bmp']
ref_img = imread(ref_img_dict[0], '0')
tar_img = imread(tar_img_dict[0], '0')

fig = plt.Figure(figsize=(2, 2), dpi=100)
a = fig.add_subplot(221)  # 添加子图:1行1列第1个
a.imshow(ref_img, cmap='gray')
b = fig.add_subplot(222)
b.imshow(tar_img, cmap='gray')

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()  # 注意show方法已经过时了,这里改用draw
canvas.get_tk_widget().pack(side=tk.TOP,  # 上对齐
                            fill=tk.BOTH,  # 填充方式
                            expand=tk.YES)  # 随窗口大小调整而调整

root.mainloop()
