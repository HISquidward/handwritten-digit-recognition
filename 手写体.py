import cv2
from tkinter import *
import cv2 as cv
import numpy as np
import tkinter as tk
import tkinter.filedialog
from PIL import Image, ImageTk
import time


# 反相灰度图，将黑白阈值颠倒
def accessPiexl(img):
    print(type(img))
    height = img.shape[0]
    width = img.shape[1]
    for i in range(height):
       for j in range(width):
           img[i][j] = 255 - img[i][j]
    return img


# 反相二值化图像
def accessBinary(img, threshold=220):
    img = accessPiexl(img)
    # 边缘膨胀，不加也可以
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    _, img = cv2.threshold(img, threshold, 0, cv2.THRESH_TOZERO)
    return img


# 根据长向量找出顶点
def extractPeek(array_vals, min_vals=10, min_rect=20):
    extrackPoints = []
    startPoint = None
    endPoint = None
    for i, point in enumerate(array_vals):
        if point > min_vals and startPoint == None:
            startPoint = i
        elif point < min_vals and startPoint != None:
            endPoint = i

        if startPoint != None and endPoint != None:
            extrackPoints.append((startPoint, endPoint))
            startPoint = None
            endPoint = None

    # 剔除一些噪点
    for point in extrackPoints:
        if point[1] - point[0] < min_rect:
            extrackPoints.remove(point)
    return extrackPoints


# 寻找边缘，返回边框的左上角和右下角（利用直方图寻找边缘算法（需行对齐））
def findBorderHistogram(path):
    borders = []
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = accessBinary(img)
    # 行扫描
    hori_vals = np.sum(img, axis=1)
    hori_points = extractPeek(hori_vals)
    # 根据每一行来扫描列
    for hori_point in hori_points:
        extractImg = img[hori_point[0]:hori_point[1], :]
        vec_vals = np.sum(extractImg, axis=0)
        vec_points = extractPeek(vec_vals, min_rect=0)
        for vect_point in vec_points:
            border = [(vect_point[0], hori_point[0]), (vect_point[1], hori_point[1])]
            borders.append(border)
    return borders


# 显示结果及边框
def showResults(path, borders, results=None):
    img = cv2.imread(path)
    # 绘制
    print(img.shape)
    for i, border in enumerate(borders):
        cv2.rectangle(img, border[0], border[1], (0, 0, 255))
        if results:
            cv2.putText(img, str(results[i]), border[0], cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 1)
        # cv2.circle(img, border[0], 1, (0, 255, 0), 0)
    return img


# 根据边框转换为MNIST格式
def transMNIST(path, borders, size=(28, 28)):
    imgData = np.zeros((len(borders), size[0], size[0], 1), dtype='uint8')
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = accessBinary(img)
    for i, border in enumerate(borders):
        borderImg = img[border[0][1]:border[1][1], border[0][0]:border[1][0]]
        # 根据最大边缘拓展像素
        extendPiexl = (max(borderImg.shape) - min(borderImg.shape)) // 2
        targetImg = cv2.copyMakeBorder(borderImg, 7, 7, extendPiexl + 7, extendPiexl + 7, cv2.BORDER_CONSTANT)
        targetImg = cv2.resize(targetImg, size)
        targetImg = np.expand_dims(targetImg, axis=-1)
        imgData[i] = targetImg
    return imgData


# 预测手写数字
def predict(modelpath, imgData):
    from keras import models
    my_mnist_model = models.load_model(modelpath)
    # print(my_mnist_model.summary())
    img = imgData.astype('float32') / 255
    results = my_mnist_model.predict(img)
    result_number = []
    for result in results:
        result_number.append(np.argmax(result))
    return result_number


def Showimage(imgCV_in, canva, layout="null"):
    """
    Showimage()是一个用于在tkinter的canvas控件中显示OpenCV图像的函数。
    使用前需要先导入库
    import cv2 as cv
    from PIL import Image,ImageTktkinter
    并注意由于响应函数的需要，本函数定义了一个全局变量 imgTK，请不要在其他地方使用这个变量名!
    参数：
    imgCV_in：待显示的OpenCV图像变量
    canva：用于显示的tkinter canvas画布变量
    layout：显示的格式。可选项为：
        "fill"：图像自动适应画布大小，并完全填充，可能会造成画面拉伸
        "fit"：根据画布大小，在不拉伸图像的情况下最大程度显示图像，可能会造成边缘空白
        给定其他参数或者不给参数将按原图像大小显示，可能会显示不全或者留空
    """
    global imgTK
    canvawidth = int(canva.winfo_reqwidth())
    canvaheight = int(canva.winfo_reqheight())

    sp = imgCV_in.shape
    cvheight = sp[0]  # height(rows) of image
    cvwidth = sp[1]  # width(colums) of image
    if (layout == "fill"):
        imgCV = cv.resize(imgCV_in, (canvawidth, canvaheight), interpolation=cv.INTER_AREA)
    elif(layout == "fit"):
        if (float(cvwidth/cvheight) > float(canvawidth/canvaheight)):
            imgCV = cv.resize(imgCV_in, (canvawidth, int(canvawidth*cvheight/cvwidth)), interpolation=cv.INTER_AREA)
        else:
            imgCV = cv.resize(imgCV_in, (int(canvaheight*cvwidth/cvheight), canvaheight), interpolation=cv.INTER_AREA)
    else:
        imgCV = imgCV_in

    imgCV = cv2.merge([imgCV, imgCV, imgCV])

    imgCV2 = cv.cvtColor(imgCV, cv.COLOR_BGR2RGBA) # 转换颜色从BGR到RGBA

    current_image = Image.fromarray(imgCV2)#将图像转换成Image对象
    imgTK = ImageTk.PhotoImage(image=current_image)#将image对象转换为imageTK对象
    canva.create_image(0, 0, anchor=NW, image=imgTK)


def choose_file():                                                      # 1号图片选择
    selectfilename = tk.filedialog.askopenfilename(title='选择图片1')
    e.set(selectfilename)   # 文件名返回给全局变量e


def open_file():                                                        # 模型选择
    file_second = tk.filedialog.askopenfilename(title='选择模型')
    a.set(file_second)


def BW_img(e_entry):
    path = e_entry.get()
    # path = path.replace("/", "\\")
    print(path)
    img = cv2.imread(path, 0)
    img = accessBinary(img)
    # img = accessBinary(img)
    Showimage(img, canva, "fill")


def Frame(e_entry, results=None):
    path = e_entry.get()
    # path = path.replace("/", "\\")
    borders = findBorderHistogram(path)
    img = cv2.imread(path)
    # 绘制
    print(img.shape)
    for i, border in enumerate(borders):
        cv2.rectangle(img, border[0], border[1], (0, 0, 255))
        if results:
            cv2.putText(img, str(results[i]), border[0], cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 1)
        # cv2.circle(img, border[0], 1, (0, 255, 0), 0)
    Showimage(img, canva, "fill")


def Cut_img(e_entry):
    path = e_entry.get()
    borders = findBorderHistogram(path)
    imgData = transMNIST(path, borders)

    for i, img in enumerate(imgData):
        Showimage(img, canva1, "fill")
        time.sleep(0.5)
        root.update()


def Prd(e_entry, a_entry):
    path = e_entry.get()
    model = a_entry.get()
    borders = findBorderHistogram(path)
    imgData = transMNIST(path, borders)
    results = predict(model, imgData)
    Showimage(showResults(path, borders, results), canva2, "fill")
    root.update()


# 图片切片保存
# path = 'D:/Pycharm_Prj/opencv learning/num5.png'
# print(path)
# borders = findBorderHistogram(path)
# imgData = transMNIST(path, borders)
# for i, img in enumerate(imgData):
    # cv2.imshow('test', img)
    # cv2.waitKey(0)
    # name = './num/test_' + str(i) + '.jpg'
    # cv2.imwrite(name, img)


if __name__ == '__main__':
    root = tk.Tk()
    root.geometry('1000x1000')  # 窗口尺寸
    root.title('MNIST 手写数字识别')  # 窗口标题
    root.resizable(False, False)

    global e, a
    e = tk.StringVar()
    a = tk.StringVar()

    e_entry = tk.Entry(root, width=68, textvariable=e)
    a_entry = tk.Entry(root, width=68, textvariable=a)

    canva = Canvas(root, width=800, height=300, bg="gray")
    canva.pack()
    canva.place(x=100, y=0)

    canva1 = Canvas(root, width=200, height=200, bg="gray")
    canva1.pack()
    canva1.place(x=100, y=600)

    canva2 = Canvas(root, width=800, height=300, bg="gray")
    canva2.pack()
    canva2.place(x=100, y=300)

    # 图片选择1按钮定义
    sumbit_btn = tk.Button(root, text="选择图片", bg='blue', command=choose_file)
    sumbit_btn.place(x=1, y=30)

    # 图片选择2按钮定义
    sumbit2_btn = tk.Button(root, text="选择模型", bg='red', command=open_file)
    sumbit2_btn.place(x=1, y=70)

    # 滤波按钮定义
    EPF_btn = tk.Button(root, text='二值化', bg='yellow', command=lambda: BW_img(e_entry))
    EPF_btn.place(x=1, y=120)

    EPF2_btn = tk.Button(root, text='字符定位', bg='yellow', command=lambda: Frame(e_entry))
    EPF2_btn.place(x=1, y=170)

    EPF3_btn = tk.Button(root, text='字符分割', bg='yellow', command=lambda: Cut_img(e_entry))
    EPF3_btn.place(x=1, y=220)

    EPF4_btn = tk.Button(root, text='预测', bg='yellow', command=lambda: Prd(e_entry, a_entry))
    EPF4_btn.place(x=1, y=270)

    root.mainloop()
