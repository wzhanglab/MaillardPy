
# -*- coding:UTF-8 -*-
from bs4 import BeautifulSoup
import requests
import json
import time

zs = 0
zys = 50


def ks(html):
    soup = BeautifulSoup(html.content, 'lxml')
    dv1 = soup.select('.table_yjfx')
    dv2 = BeautifulSoup(str(dv1), 'lxml')
    dv3 = dv2.select('tr')
    dv3 = dv3[2:-1]
    aa = []
    zdsl = []
    j = 1
    for i in range(len(dv3)):
        tmp = str(dv3[i])
        if tmp.find('background:#EFEFEF') == -1:
            j = j + 1
            if i == int(len(dv3)-1):
                zdsl.append(j)
            continue
        if i > 0:
            zdsl.append(j)
            j = 1
    print(zdsl)
    j = 0
    for i in range(0, len(zdsl)):
        bb = {}
        sjcd = zdsl[i]
        # print(sjcd)
        td1 = dv3[j].select('td')
        bb['author'] = td1[0].get_text()
        bb['organ'] = td1[1].get_text()
        bb['jine'] = td1[2].get_text()
        bb['xmbh'] = td1[3].get_text()
        bb['xmlx'] = td1[4].get_text()
        bb['ssxb'] = td1[5].get_text()
        bb['year'] = td1[6].get_text()
        td2 = dv3[j+1].select('td')
        bb['title'] = td2[1].get_text()
        td3 = dv3[j+2].select('td')
        bb['xkfl'] = td3[1].get_text()
        td4 = dv3[j+3].select('td')
        bb['xkdm'] = td4[1].get_text()
        td5 = dv3[j+4].select('td')
        bb['zxsj'] = td5[1].get_text()
        if sjcd == 5:
            bb['keycn'] = ''
            bb['keyen'] = ''
            bb['discript'] = ''
        if sjcd == 6:
            td6 = dv3[j+5].select('td')
            bb['keycn'] = td6[1].get_text()
            bb['keyen'] = ''
            bb['discript'] = ''
        if sjcd == 7:
            td6 = dv3[j+5].select('td')
            bb['keycn'] = td6[1].get_text()
            td7 = dv3[j+6].select('td')
            xmbt = td7[0].get_text()
            # print(td7)
            if xmbt == '结题摘要':
                bb['keyen'] = ''
                bb['discript'] = td7[1].get_text()
            else:
                bb['keyen'] = td7[1].get_text()
                bb['discript'] =''
        if sjcd == 8:
            td6 = dv3[j+5].select('td')
            bb['keycn'] = td6[1].get_text()
            td7 = dv3[j+6].select('td')
            bb['keyen'] = td7[1].get_text()
            td8 = dv3[j+7].select('td')
            bb['descript'] = td8[1].get_text()
        j = j + sjcd
        aa.append(bb)
    return aa


def data_out(data):
    #这里写成一个方法好处是，在写入文本的时候就在这里写
    fo = open("D:\data.txt", "a+") #这里注意重新写一个地址
    #for i,e in enumerate(data):
    fo.write(data)
        #print '第%d个，title：%s' % (i,e)
    # 关闭打开的文件
    fo.close()

url = 'http://www.letpub.com.cn/nsfcfund_search.php?mode=advanced&datakind=list&page=&name=&person=&no=&company=&addcomment_s1=C&&addcomment_s2=C20&addcomment_s3=&addcomment_s4=&money1=&money2=&startTime=1997&endTime=2019&subcategory=&searchsubmit=true&'
userAgent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36"
header = {
    "Host": "www.letpub.com.cn",
    "User-Agent": userAgent
}
for yeshu in range(1, 20):
    print('开始获取第'+str(yeshu)+'页内容')
    url1 = url+'currentpage='+str(yeshu)+'#fundlisttable'
    # print(url1)
    html = requests.post(url1, headers=header)
    html.encoding = html.apparent_encoding
    # print(html.content)
    data = ks(html)
    data_out(str(data))
    time.sleep(10)
