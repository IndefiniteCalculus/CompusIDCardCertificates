import re

str = '二 、;\n' \
      '学 二 : 2017210906\n' \
      '姓 名: 江 笑语\n' \
      '所 属 学 院 : 光电 工程 学 院\n' \
      '发 卡 日 期 : 2020-10-09\n' \
      '有 读 一 认 三 现 : 1652157\n' \
      ',PT'


def str_split(str):
    str = str.replace(" ", "")
    s = re.findall(":(.*?)\n", str)
    for i in s:
        if '-' in i:
            s.remove(i)

    return s
if __name__ == '__main__':
    stu_id = '2017210906'
    uid = '165215712345671654438'
    name = "光电江笑语自动化张子澳"
    matcher = re.compile('江笑语|刘星宇')
    span = matcher.search(name).span()
    get = name[span[0]:span[1]]
    pass