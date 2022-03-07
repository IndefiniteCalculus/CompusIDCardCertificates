from SystemConponents import SQLTool as sql

stu_id = '2017210827'
con = sql.link()
result = sql.get_one_person(con, stu_id)
sql.shut(con)

print(result)