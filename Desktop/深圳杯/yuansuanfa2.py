import os.path
from cv2 import dct
import os.path
from pywt import dwt2
import cv2
import numpy
import numpy as np
from numpy.linalg import svd
import sys
import multiprocessing
import warnings
# 多线程设置
if sys.platform != 'win32':
    multiprocessing.set_start_method('fork')

class CommonPool(object):
    def map(self, func, args):
        return list(map(func, args))

class AutoPool(object):
    def __init__(self, mode, processes):
        if mode == 'multiprocessing' and sys.platform == 'win32':
            warnings.warn('multiprocessing not support in windows, turning to multithreading')
            mode = 'multithreading'
        self.mode = mode
        self.processes = processes
        if mode == 'vectorization':
            pass
        elif mode == 'cached':
            pass
        elif mode == 'multithreading':
            from multiprocessing.dummy import Pool as ThreadPool
            self.pool = ThreadPool(processes=processes)
        elif mode == 'multiprocessing':
            from multiprocessing import Pool
            self.pool = Pool(processes=processes)
        else:  # common
            self.pool = CommonPool()
    def map(self, func, args):
        return self.pool.map(func, args)


def read_img(filename):
    assert os.path.exists(filename), filename + '不存在'
    return cv2.imread(filename, flags=cv2.IMREAD_UNCHANGED)


def random_strategy1(seed, size, block_shape):
    return np.random.RandomState(seed) \
        .random(size=(size, block_shape)) \
        .argsort(axis=1)

class text_core_function:
    def __init__(self, password=1, mode='str', encoding='gbk', length_ran=False, out_of_place=True, ratio=1):  # gbk编码省空间
        self.fast_mode = False
        self.wm_size = 0
        self.password = password
        self.img = None
        self.img_YUV = None  # 采用YUV通道,默认不透明图像
        self.block_shape = np.array([4, 4])  # 参考文献里采用的8x8，这里采用4x4
        self.ll, self.hvd, = [np.array([])] * 3, [np.array([])] * 3  # ll为低频域，hvd是其他三个细节部分
        self.ll_block = [np.array([])] * 3  # 每个 channel 存一个四维 array，代表四维分块后的结果
        self.ll_part = [np.array([])] * 3  # 四维分块后，有时因不整除而少一部分，self.ca_part 是少这一部分的 self.ca
        assert mode == 'str', '暂时不支持字符串以外的水印'
        self.mode = mode
        self.wm_content = None  # 水印内容
        self.wm_bit = None  # 字节水印
        self.encoding = encoding
        self.inited = False
        self.ll_part = [np.array([])] * 3  # 四维分块后，有时因不整除而少一部分，self.ca_part 是少这一部分的 self.ca
        self.d1, self.d2 = 36, 20  # d1/d2 越大鲁棒性越强,但输出图片的失真越大
        self.pool = AutoPool(mode='common', processes=None)
        self.length_ran = length_ran
        self.out = out_of_place
        self.ratio = ratio

    def read_img_to_arr(self, img: numpy.ndarray):
        self.img_shape = img.shape[:2]
        self.img_YUV = cv2.copyMakeBorder(cv2.cvtColor(img, cv2.COLOR_BGR2YUV),
                                          0, img.shape[0] % 2, 0, img.shape[1] % 2,
                                          cv2.BORDER_CONSTANT, value=(0, 0, 0))
        self.ll_shape = [i // 2 for i in self.img_shape]
        self.ll_block_shape = (
            self.ll_shape[0] // self.block_shape[0], self.ll_shape[1] // self.block_shape[1], self.block_shape[0],
            self.block_shape[1])
        # 步长，跨越维度和元素需要的字节数
        strides = 4 * np.array([self.ll_shape[1] * self.block_shape[0], self.block_shape[1], self.ll_shape[1], 1])
        for channel in range(3):
            # 对每个通道进行小波变换
            self.ll[channel], self.hvd[channel] = dwt2(self.img_YUV[:, :, channel], 'haar')
            # 使用滑动窗口把ll部分一分为四
            self.ll_block[channel] = np.lib.stride_tricks.as_strided(self.ll[channel].astype(np.float32),
                                                                     self.ll_block_shape, strides)

    def init_block_index(self):
        self.block_num = self.ll_block_shape[0] * self.ll_block_shape[1]
        assert self.wm_size < self.block_num, IndexError(
            '最多可嵌入{}kb信息，多于水印的{}kb信息，溢出'.format(self.block_num / 1000, self.wm_size / 1000))
        # self.part_shape 是取整后的ca二维大小,用于嵌入时忽略右边和下面对不齐的细条部分。
        self.part_shape = self.ll_block_shape[:2] * self.block_shape
        self.block_index = [(i, j) for i in range(self.ll_block_shape[0]) for j in range(self.ll_block_shape[1])]



def one_dim_kmeans(inputs):
    threshold = 0
    e_tol = 10 ** (-6)
    center = [inputs.min(), inputs.max()]  # 1. 初始化中心点
    for i in range(300):
        threshold = (center[0] + center[1]) / 2
        is_class01 = inputs > threshold  # 2. 检查所有点与这k个点之间的距离，每个点归类到最近的中心
        center = [inputs[~is_class01].mean(), inputs[is_class01].mean()]  # 3. 重新找中心点
        if np.abs((center[0] + center[1]) / 2 - threshold) < e_tol:  # 4. 停止条件
            threshold = (center[0] + center[1]) / 2
            break
    is_class01 = inputs > threshold
    return is_class01

class extractor(text_core_function):
    def __init__(self, encoding='gbk'):
        super().__init__(encoding=encoding)
        self.sss = []

    def extract_form_file(self, wm_shape=5224, filename=None):
        assert os.path.exists(filename), '文件不存在'
        self.ex_img = read_img(filename)
        wm_avg = self.extract_with_kmeans(img=self.ex_img, wm_shape=wm_shape)
        wm = self.extract_decrypt(wm_avg=wm_avg)
        byte = ''.join(str((i >= 0.5) * 1) for i in wm)
        wm = bytes.fromhex(hex(int(byte, base=2))[2:]).decode(self.encoding, errors='replace')
        print(wm.replace('$$', '\n'))
        return wm.replace('$$', '\n')

    def one_block_get_wm(self, args):
        block, shuffler = args
        block_dct_shuffled = dct(block).flatten()[shuffler].reshape(self.block_shape)
        u, s, v = svd(block_dct_shuffled)
        self.sss.append(s[0] % self.d1)
        wm = (s[0] % self.d1 > self.d2 / 2) * 1
        if self.d2:
            tmp = (s[1] % self.d2 > self.d2 / 2) * 1
            wm = (wm * 3 + tmp * 1) / 4
        return wm

    def extract_bit_from_img(self, img):
        self.read_img_to_arr(img=img)
        self.init_block_index()
        wm_block_bit = np.zeros(shape=(3, self.block_num))
        self.idx_shuffle = random_strategy1(seed=self.password,
                                            size=self.block_num,
                                            block_shape=self.block_shape[0] * self.block_shape[1],  # 16
                                            )

        for channel in range(3):
            wm_block_bit[channel, :] = self.pool.map(self.one_block_get_wm,
                                                     [(self.ll_block[channel][self.block_index[i]],
                                                       self.idx_shuffle[i])
                                                      for i in range(self.block_num)])

        return wm_block_bit

    def extract_avg(self, wm_block_bit):
        # 对循环嵌入+3个 channel 求平均
        wm_avg = np.zeros(shape=self.wm_size)
        for i in range(self.wm_size):
            wm_avg[i] = wm_block_bit[:, i::self.wm_size].mean()
        return wm_avg

    def extract(self, img, wm_shape):
        self.wm_size = np.array(wm_shape).prod()
        # 提取每个分块埋入的 bit：
        wm_block_bit = self.extract_bit_from_img(img=img)
        # 做平均：
        wm_avg = self.extract_avg(wm_block_bit)
        return wm_avg

    def extract_with_kmeans(self, img, wm_shape):
        wm_avg = self.extract(img=img, wm_shape=wm_shape)
        return one_dim_kmeans(wm_avg)

    def extract_decrypt(self, wm_avg):
        wm_index = np.arange(self.wm_size)
        np.random.RandomState(self.password).shuffle(wm_index)
        wm_avg[wm_index] = wm_avg.copy()
        return wm_avg

if __name__ == "__main__":
    a = extractor(encoding='utf-8')
    a.extract_form_file(filename='test.jpg')



