import pymysql
import re
# 远程登陆数据库
def link():
    con = pymysql.Connect(
        host = '49.234.81.155',  # 外网地址 (数据库管理中查看)
        port = 3306,  # 外网端口 (数据库管理中查看)
        user = 'root',  # 账号 (初始化的账号)
        passwd = 'abc12345654321',  # 密码 (初始化的密码)
        db = 'Carded_compus'  # 数据库名称
    )
    return con

# 关闭数据库
def shut(con):
    con.close()
def exe(con,sql:str):
    cur = con.cursor()
    data = []
    try:
        # sql1 = 'insert into student_info values("2017210827","刘星宇","计算机科学与技术学院","1651542")'
        # cur.execute(sql1)
        # con.commit()
        cur.execute(sql)
        result = cur.fetchall()
        data.append(result)
    except pymysql.Error as e:
        print("Error %d: %s" % (e.args[0], e.args[1]))

    cur.close()
    return data
# 获取本次的游标
def get_one_person(student_id):
    con = link()
    sql = 'select * from student_info where stu_id  regexp '+student_id
    result = exe(con, sql)
    con.close()
    if len(result) == 1:
        return result[0][0]
    else:
        return result[0][0]

def get_distinct_name(pattern=True):
    con = link()
    sql = 'select distinct name from student_info'
    got = exe(con,sql)
    con.close()
    result = []
    for item in got[0]:
        result.append(item[0])
    if pattern == True:
        return get_pattern(result, tolerate=False)
    else:
        return result

def get_distinct_stuid(pattern=True):
    '''-->((str),(str),(str))'''
    con = link()
    sql = 'select distinct stu_id from student_info'
    got = exe(con, sql)
    con.close()
    result = []
    for item in got[0]:
        result.append(item[0])
    if pattern == True:
        return get_pattern(result,  tolerate=False)
    else:
        return result

def get_distinct_college(pattern=True):
    con = link()
    sql = 'select distinct college from student_info'
    got = exe(con, sql)
    con.close()
    result = []
    for item in got[0]:
        result.append(item[0])
    if pattern == True:
        return get_pattern(result,  tolerate=True)
    else:
        return result

def get_distinct_uid(pattern=True):
    con = link()
    sql = 'select distinct user_id from student_info'
    got = exe(con, sql)
    con.close()
    result = []
    for item in got[0]:
        result.append(item[0])
    if pattern == True:
        return get_pattern(result, tolerate=False)
    else:
        return result

def get_pattern(result, tolerate = True):
    pattern = ''
    if tolerate == True:
        split1,split2 = '[',']'
    else:
        split1,split2 = '(',')'
    for college_idx in range(len(result)):
        pattern += split1+ result[college_idx]
        if tolerate == True:
            # 为[]类匹配添加匹配次数
            pattern += split2 + '{' + str(len(result[college_idx]) // 2) + ',' + str(
                len(result[college_idx])) + '}'
            if len(result) - 1 == college_idx:
                pass
            else:
                pattern += '|'
        else:
            # 为()类匹配添加匹配次数
            if len(result) - 1 == college_idx:
                pattern += split2 + '{1}'
            else:
                pattern += split2 + '{1}'+'|'
    return pattern
if __name__ == '__main__':

    person_info = get_one_person('2017214715')
    names = get_distinct_name()
    college = get_distinct_college()
    # uid = get_distinct_uid()
    # uid = re.compile(uid)
    # result = re.search(uid, '统一认证码:1654438')
    pass