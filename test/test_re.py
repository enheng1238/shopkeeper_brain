import re

md_content = "这里有一张图 ![示意图](images/demo(1).png) 再来一张 ![图片](images/demo(1).png)"
image_name = "demo(1).png"

# re.escape() 是 Python 标准库 re 模块中的函数，用于转义字符串中的正则表达式特殊字符。
# r"\!\[.*?\]\(.*? " + re.escape(image_name) + r".*?\)"

#
# # 编译正则（不变） 正则中有两类字符 元字符 . * ? ! [a-z] {} 字面字符 a  png
rep = re.compile(r"\!\[.*?\]\(.*?" + re.escape(image_name) + r".*?\)")
print("rep",rep)

#
# # 1. search：全文找【第一个】匹配 → 返回 match 对象
# match_obj = rep.search(md_content)
# print("1. search 结果：", match_obj.group() if match_obj else "未找到", type(match_obj))

# # 2. match：只从【开头】匹配 → 这里开头不是图片语法，必找不到
# match_first = rep.match(md_content)
# print("2. match 结果：", "找到" if match_first else "未找到（只匹配开头）")
#
# 3. findall：找【所有】匹配 → 返回字符串列表
# all_list = rep.findall(md_content)
# print("3. findall 列表：", all_list, type(all_list))

# # 4. finditer：找【所有】匹配 → 返回 match 对象（你要的“search 匹配所有”）
# print("4. finditer 遍历所有：")
for m in rep.finditer(md_content):
    print("  - 内容：", m.group(), " | 位置：", m.span(), type(m))

# () 和 ?  贪婪匹配
# text = "A123B456B"
# # 1. 用 .*  → 贪婪（吃到最后）
# print(re.findall(r"A(.*)B", text))  # 输出：['123B456']
# # 2. 用 .*? → 非贪婪（找到第一个就停）
# print(re.findall(r"A(.*?)B", text)) # 输出：['123']