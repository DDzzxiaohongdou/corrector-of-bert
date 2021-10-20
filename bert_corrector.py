# -*- coding: utf-8 -*-
import operator
import sys
import time
from pycorrector.transformers import pipeline
sys.path.append('../..')
import config
from utils.text_utils import is_chinese_string, convert_to_unicode
from utils.logger import logger
from corrector import Corrector

class BertCorrector(Corrector):
    def __init__(self, bert_model_dir=config.bert_model_dir):
        super(BertCorrector, self).__init__()
        self.name = 'bert_corrector'
        t1 = time.time()
        self.model = pipeline('fill-mask', model=bert_model_dir, tokenizer=bert_model_dir)
        if self.model:
            self.mask = self.model.tokenizer.mask_token
            logger.debug('Loaded bert model: %s, spend: %.3f s.' % (bert_model_dir, time.time() - t1))

    def bert_correct(self, text):
        text_new = ''
        details = []
        self.check_corrector_initialized()
        # 编码统一，utf-8 to unicode
        text = convert_to_unicode(text)
        # 长句切分为短句
        blocks = self.split_2_short_text(text, include_symbol=True)  # 获得短句的及其起始位置

        for blk, start_idx in blocks:
            blk_new = ''
            for idx, s in enumerate(blk):

                # 对非中文的错误不做处理
                if is_chinese_string(s):
                    sentence_lst = list(blk_new + blk[idx:])
                    sentence_lst[idx] = self.mask
                    sentence_new = ''.join(sentence_lst)
                    predicts = self.model(sentence_new)
                    top_tokens = []
                    for p in predicts:
                        #获取token对应值内容
                        token_id = p.get('token', 0)
                        token_str = self.model.tokenizer.convert_ids_to_tokens(token_id)
                        top_tokens.append(token_str)  # 得到可能替换的top词

                    #如果top词不为空，并且原字符不在top词列表里
                    if top_tokens and (s not in top_tokens):
                        # 取得所有候选词
                        candidates = self.generate_items(s)

                        #如果候选词不为空
                        if candidates:
                            for token_str in top_tokens:
                                #对top词循环，并且当top字在候选词库中时，details列表把原字与top字加进来，并且用top字替换原字
                               if token_str in candidates:
                                    details.append([s, token_str, start_idx + idx, start_idx + idx + 1])
                                    s = token_str
                                    break
                                #如果不在，自然不变化
                #每个单句的拼接
                blk_new += s
            #复合句的拼接
            text_new += blk_new
        details = sorted(details, key=operator.itemgetter(2))
        return text_new, details


if __name__ == "__main__":
    d = BertCorrector()
    error_sentence = ['服务员，这个商品还诱惑吗？',
                      '必够吗',
                      '必狗码',
                      '必够码',
                      '必购码',
                      '必狗码',
                      '成成分现金',
                      '京喜',
                      '京品电脑',
                      '京东京造',
                      '精品电脑',
                      '京遵达',
                      '京豆',
                      '京痘',
                      '京东有礼',
                      '京东有利',
                      '省省卡',
                      '校园贷',
                      '京享值',
                      '企享值',
                      '主子账户',
                      '一分购',
                      '一份购',
                      '东东农场',
                      '京喜拼拼',
                      '惊喜',
                      '京西',
                      '京喜',
                      '京东E卡',
                      '云闪付',
                      '膨胀金',
                      '疯控',
                      '风控',
                      '沸腾之夜',
                      '博阅打卡',
                      '首购礼金',
                      '惠购卡',
                      '回购卡',
                      '阿卡索外教网',
                      '阿卡索外交网',
                      '外交网',
                      '月卡',
                      '海囤全球',
                      '推荐有礼',
                      '推荐有利',
                      '待打款',
                      '会员食力派',
                      '异动',
                      '移动',
                      '易支付',
                      '京选优品',
                      '优惠卷',
                      '京东极速版',
                      '精准通',
                      '嗨回收热线',
                      '自营',
                      '砸京蛋',
                      '更换主体',
                      '粉丝专享券',
                      '药京采',
                      '食力街',
                      '加购有礼',
                      '加购有利',
                      '京东省钱包',
                      '京橙好店',
                      '定期购',
                      '定期狗',
                      '京车汇',
                      '家庭号',
                      '家电回收',
                      '东东爱消除',
                      '全球购',
                      '全球狗']
    '''
    error_sentence = ['万一坏了呢',
                      '三个x穿多少斤',
                      '三个加能穿多少斤',
                      '三个叉的是多少码',
                      '三个月用什么号',
                      '三个灯一起闪是什么故障',
                      '三人位的长多少',
                      '三加穿多少斤',
                      '三十九码谢L码会小嘛',
                      '三十太多了',
                      '三只大鳄龟',
                      '三天也够了',
                      '三天了没有动',
                      '三天就慢了',
                      '三天物流没有动',
                      '三天都不止',
                      '三岁宝宝什么码',
                      '三岁宝宝能吃不',
                      '三年级用会小吗',
                      '三挡调光什么意思',
                      '三斤大果多少个啊',
                      '三星7edge',
                      '三星电视怎么安装其他app',
                      '三瓶要洗多久',
                      '三盒的话多少钱',
                      '三米九的柜子能做不',
                      '三米六的杆稍']
    '''
    import datetime
    for sent in error_sentence:

        tic = datetime.datetime.now()
        corrected_sent, err = d.bert_correct(sent)
        end = datetime.datetime.now()
        print("original sentence:{} => {}, err:{}".format(sent, corrected_sent, err))
        #print("耗时 %s seconds" % (end - tic))
