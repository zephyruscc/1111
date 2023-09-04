import copy
import os.path
from pywt import dwt2, idwt2
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

def random_strategy2(seed, size, block_shape):
    one_line = np.random.RandomState(seed) \
        .random(size=(1, block_shape)) \
        .argsort(axis=1)
    return np.repeat(one_line, repeats=size, axis=0)

def string_to_binary(s, encoding='utf-8'):
    # 首先，使用指定的编码将字符串转化为字节
    byte_sequence = s.encode(encoding)
    # 然后，将每一个字节转化为它的二进制表示，并连接在一起
    return ''.join(format(byte, '08b') for byte in byte_sequence)

def binary_to_string(binary_str, encoding='utf-8'):
    # 分割二进制字符串为每8位一组
    byte_strings = [binary_str[i:i + 8] for i in range(0, len(binary_str), 8)]
    # 将每组二进制字符串转换为其对应的字节
    byte_sequence = bytes([int(byte_str, 2) for byte_str in byte_strings])
    # 使用指定的编码将字节序列解码为字符串
    return byte_sequence.decode(encoding)

def to_limit_length(text, hopped_length, ratio=1):
    lenth = int(hopped_length/(16*ratio))
    return text[:lenth]

def replace(text: str):
    return text.replace('\n', '$$').replace(' ', '')

class text_core_function:
    def __init__(self, password=1, mode='str', encoding='gbk', length_ran=False, out_of_place=True, ratio=1):  # gbk编码省空间
        self.fast_mode = False
        self.wm_size = 0
        self.password = password
        self.img = None
        self.img_YUV = None  # 采用YUV通道,默认不透明图像
        self.block_shape = np.array([4, 4])  # 设置分块大小
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

    def init_emb_func(self, filename, wm_content):
        self.img = read_img(filename).astype(np.float32)
        self.read_img_to_arr(self.img)
        self.wm_content = wm_content
        self.wm_cont_func()
        self.init_block_index()
        self.inited = True
        return self.wm_size, self.wm_content

    # 仅提供字符串嵌入
    def wm_cont_func(self):
        if self.out:
            self.wm_content = replace(self.wm_content)
        if self.mode == 'str':
            byte = bin(int(self.wm_content.encode(self.encoding).hex(), base=16))[2:]
            self.wm_bit = (np.array(list(byte)) == '1')
        self.block_num = self.ll_block_shape[0] * self.ll_block_shape[1]

        if self.length_ran and self.wm_bit.size > self.block_num:
            print(self.wm_bit.size, self.block_num)
            self.wm_content = to_limit_length(self.wm_content, self.block_num, self.ratio)
            print('舍弃后字符长度', len(self.wm_content))
            limit = string_to_binary(self.wm_content, self.encoding)
            print('嵌入长度', len(limit))
            print("已经舍去后面{}b信息".format(self.wm_bit.size - len(limit)))
            self.wm_cont_func()
            return
        np.random.RandomState(self.password).shuffle(self.wm_bit)
        self.wm_size = self.wm_bit.size


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

    def embed_func(self):
        assert self.inited is True, "未初始化，请使用init_class_func方法"
        embed_ca = copy.deepcopy(self.ll)
        embed_YUV = [np.array([])] * 3
        self.idx_shuffle = random_strategy1(self.password, self.block_num, self.block_shape[0] * self.block_shape[1])
        for channel in range(3):
            tmp = self.pool.map(self.block_add_wm,
                                [(self.ll_block[channel][self.block_index[i]], self.idx_shuffle[i], i)
                                 for i in range(self.block_num)])
            for i in range(self.block_num):
                self.ll_block[channel][self.block_index[i]] = tmp[i]
            # 4维分块变回2维
            self.ll_part[channel] = np.concatenate(np.concatenate(self.ll_block[channel], 1), 1)
            # 4维分块时右边和下边不能整除的长条保留，其余是主体部分，换成 embed 之后的频域的数据
            embed_ca[channel][:self.part_shape[0], :self.part_shape[1]] = self.ll_part[channel]
            # 逆变换回去
            embed_YUV[channel] = idwt2((embed_ca[channel], self.hvd[channel]), "haar")
        # 合并3通道
        embed_img_YUV = np.stack(embed_YUV, axis=2)
        # 之前如果不是2的整数，增加了白边，这里去除掉
        embed_img_YUV = embed_img_YUV[:self.img_shape[0], :self.img_shape[1]]
        embed_img = cv2.cvtColor(embed_img_YUV, cv2.COLOR_YUV2BGR)
        embed_img = np.clip(embed_img, a_min=0, a_max=255)
        return embed_img

    def block_add_wm(self, arg):
        return self.block_add_wm_slow(arg)

    def block_add_wm_slow(self, arg):
        # 4x4
        block, shuffler, i = arg
        # dct->(flatten->加密->逆flatten)->svd->打水印->逆svd->(flatten->解密->逆flatten)->逆dct
        wm_1 = self.wm_bit[i % self.wm_size]
        block_dct = cv2.dct(block)
        # 加密（打乱顺序）
        block_dct_shuffled = block_dct.flatten()[shuffler].reshape(self.block_shape)

        u, s, v = svd(block_dct_shuffled)
        s[0] = (s[0] // self.d1 + 1 / 4 + 1 / 2 * wm_1) * self.d1
        if self.d2:
            s[1] = (s[1] // self.d2 + 1 / 4 + 1 / 2 * wm_1) * self.d2

        block_dct_flatten = np.dot(u, np.dot(np.diag(s), v)).flatten()
        block_dct_flatten[shuffler] = block_dct_flatten.copy()
        return cv2.idct(block_dct_flatten.reshape(self.block_shape))

    def embed(self, filename=None, compression_ratio=None):
        embed_img = self.embed_func()
        if filename is not None:
            if compression_ratio is None:
                cv2.imwrite(filename=filename, img=embed_img)
            elif filename.endswith('.jpg'):
                cv2.imwrite(filename=filename, img=embed_img, params=[cv2.IMWRITE_JPEG_QUALITY, compression_ratio])
            elif filename.endswith('.png'):
                cv2.imwrite(filename=filename, img=embed_img, params=[cv2.IMWRITE_PNG_COMPRESSION, compression_ratio])
            else:
                cv2.imwrite(filename=filename, img=embed_img)
        return embed_img

    def init_block_index(self):
        self.block_num = self.ll_block_shape[0] * self.ll_block_shape[1]
        print('水印大小', self.wm_size)
        assert self.wm_size < self.block_num, IndexError(
            '最多可嵌入{}kb信息，多于水印的{}kb信息，溢出'.format(self.block_num / 1000, self.wm_size / 1000))
        print('最多可嵌入{}kb信息'.format(self.block_num / 1000))
        # self.part_shape 是取整后的ca二维大小,用于嵌入时忽略右边和下面对不齐的细条部分。
        self.part_shape = self.ll_block_shape[:2] * self.block_shape
        self.block_index = [(i, j) for i in range(self.ll_block_shape[0]) for j in range(self.ll_block_shape[1])]

    def test_info(self):
        print('低频域大小：',self.ll_shape, '图片大小：',self.img_shape,'低频域分块数：', self.ll_block_shape,'分块大小:', self.block_shape)
        data = bin(int(self.wm_content.encode(self.encoding).hex(), base=16))[2:]
        str_ = bytes.fromhex(hex(int(data, 2))[2:]).decode(self.encoding)
        self.wm_size = self.wm_bit.size

if __name__ == "__main__":
    a = text_core_function(encoding='utf-8')
    # 这里的文件即附件里的图片 注意图片名需要修改为英文
    import docx
    def read_word_document(file_path):
        doc = docx.Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text
        return text
    word_file_path = '著作权法.docx'
    text_content = read_word_document(word_file_path)
    print(text_content)
    a.init_emb_func("Bque.jpg", text_content)
    a.test_info()
    # 最终输出加密后的图片并保存
    a.embed(filename='test.jpg')