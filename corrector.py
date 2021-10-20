import config
import codecs
from utils.logger import logger
from pypinyin import lazy_pinyin
from utils.text_utils import is_chinese_string
import re
import os

class Corrector():
    def __init__(self, common_char_path=config.common_char_path,
                       same_pinyin_path=config.same_pinyin_path,
                       same_stroke_path=config.same_stroke_path,
                       word_freq_path=config.word_freq_path,):
        self.common_char_path = common_char_path
        self.same_pinyin_text_path = same_pinyin_path
        self.same_stroke_text_path = same_stroke_path
        self.word_freq_path = word_freq_path
        self.custom_confusion = {}
        self.initialized_corrector = False
        self.cn_char_set = None
        self.same_pinyin = None
        self.same_stroke = None

    @staticmethod
    def load_set_file(path):
        words = set()
        with codecs.open(path, 'r', encoding='utf-8') as f:
            for w in f:
                w = w.strip()
                if w.startswith('#'):
                    continue
                if w:
                    words.add(w)
        return words

    @staticmethod
    def load_same_pinyin(path, sep='\t'):
        """
        加载同音字
        :param path:
        :param sep:
        :return:
        """
        result = dict()
        if not os.path.exists(path):
            logger.warn("file not exists:" + path)
            return result
        with codecs.open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#'):
                    continue
                parts = line.split(sep)
                if parts and len(parts) > 2:
                    key_char = parts[0]
                    same_pron_same_tone = set(list(parts[1]))
                    same_pron_diff_tone = set(list(parts[2]))
                    value = same_pron_same_tone.union(same_pron_diff_tone)
                    if key_char and value:
                        result[key_char] = value
        return result

    @staticmethod
    def load_same_stroke(path, sep='\t'):
        """
        加载形似字
        :param path:
        :param sep:
        :return:
        """
        result = dict()
        if not os.path.exists(path):
            logger.warn("file not exists:" + path)
            return result
        with codecs.open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#'):
                    continue
                parts = line.split(sep)
                if parts and len(parts) > 1:
                    for i, c in enumerate(parts):
                        result[c] = set(list(parts[:i] + parts[i + 1:]))
        return result

    @staticmethod
    def _initialize_corrector(self):
        # chinese common char
        self.cn_char_set = self.load_set_file(self.common_char_path)
        # same pinyin
        self.same_pinyin = self.load_same_pinyin(self.same_pinyin_text_path)
        # same stroke
        self.same_stroke = self.load_same_stroke(self.same_stroke_text_path)
        self.word_freq = self.load_word_freq_dict(self.word_freq_path)

        self.initialized_corrector = True

    def check_corrector_initialized(self):
        if not self.initialized_corrector:
            self._initialize_corrector(self)

    @staticmethod
    def split_2_short_text(text, include_symbol=False):
        re_han = re.compile("([\u4E00-\u9FD5a-zA-Z0-9+#&]+)", re.U)
        result = []
        blocks = re_han.split(text)
        start_idx = 0
        for blk in blocks:
            if not blk:
                continue
            if include_symbol:
                result.append((blk, start_idx))
            else:
                if re_han.match(blk):
                    result.append((blk, start_idx))
            start_idx += len(blk)
        return result

    def get_same_pinyin(self, char):
        # 取同音字
        self.check_corrector_initialized()
        return self.same_pinyin.get(char, set())

    def get_same_stroke(self, char):
        # 取形似字
        self.check_corrector_initialized()
        return self.same_stroke.get(char, set())

    @staticmethod
    def load_word_freq_dict(path):
        """
        加载切词词典
        :param path:
        :return:
        """
        word_freq = {}
        if not os.path.exists(path):
            logger.warning('file not found.%s' % path)
            return word_freq
        with codecs.open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#'):
                    continue
                info = line.split()
                if len(info) < 1:
                    continue
                word = info[0]
                # 取词频，默认1
                freq = int(info[1]) if len(info) > 1 else 1
                word_freq[word] = freq
        return word_freq

    def known(self, words):
        """
        取得词序列中属于常用词部分
        :param words:
        :return:
        """
        return set(word for word in words if word in self.word_freq)

    def _confusion_char_set(self, c):
        return self.get_same_pinyin(c).union(self.get_same_stroke(c))

    @staticmethod
    def edit_distance_word(word, char_set):
        """
        all edits that are one edit away from 'word'
        :param word:
        :param char_set:
        :return:
        """
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in char_set]
        return set(transposes + replaces)

    def _confusion_word_set(self, word):
        confusion_word_set = set()
        candidate_words = list(self.known(self.edit_distance_word(word, self.cn_char_set)))
        for candidate_word in candidate_words:
            if lazy_pinyin(candidate_word) == lazy_pinyin(word):
                # same pinyin
                confusion_word_set.add(candidate_word)
        return confusion_word_set

    def _confusion_custom_set(self, word):
        confusion_word_set = set()
        if word in self.custom_confusion:
            confusion_word_set = {self.custom_confusion[word]}
        return confusion_word_set

    def word_frequency(self, word):
        """
        取词在样本中的词频
        :param word:
        :return:
        """
        return self.word_freq.get(word, 0)

    def generate_items(self, word, fragment=1):
        """
        生成纠错候选集
        :param word:
        :param fragment: 分段
        :return:
        """
        self.check_corrector_initialized()
        # 1字
        candidates_1 = []
        # 2字
        candidates_2 = []
        # 多于2字
        candidates_3 = []

        # same pinyin word
        candidates_1.extend(self._confusion_word_set(word))
        # custom confusion word
        candidates_1.extend(self._confusion_custom_set(word))
        # same pinyin char
        if len(word) == 1:
            # same one char pinyin
            confusion = [i for i in self._confusion_char_set(word[0]) if i]
            candidates_1.extend(confusion)
        if len(word) == 2:
            # same first char pinyin
            confusion = [i + word[1:] for i in self._confusion_char_set(word[0]) if i]
            candidates_2.extend(confusion)
            # same last char pinyin
            confusion = [word[:-1] + i for i in self._confusion_char_set(word[-1]) if i]
            candidates_2.extend(confusion)
        if len(word) > 2:
            # same mid char pinyin
            confusion = [word[0] + i + word[2:] for i in self._confusion_char_set(word[1])]
            candidates_3.extend(confusion)

            # same first word pinyin
            confusion_word = [i + word[-1] for i in self._confusion_word_set(word[:-1])]
            candidates_3.extend(confusion_word)

            # same last word pinyin
            confusion_word = [word[0] + i for i in self._confusion_word_set(word[1:])]
            candidates_3.extend(confusion_word)

        # add all confusion word list
        confusion_word_set = set(candidates_1 + candidates_2 + candidates_3)
        confusion_word_list = [item for item in confusion_word_set if is_chinese_string(item)]
        confusion_sorted = sorted(confusion_word_list, key=lambda k: self.word_frequency(k), reverse=True)
        return confusion_sorted[:len(confusion_word_list) // fragment + 1]
