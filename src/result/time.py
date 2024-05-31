from datetime import datetime

# 定义两个时间戳字符串
time_str1 = "2023-02-22 12:59:43"
time_str2 = "2023-02-22 18:04:24"

# 将时间戳字符串转换为 datetime 对象
time1 = datetime.strptime(time_str1, "%Y-%m-%d %H:%M:%S")
time2 = datetime.strptime(time_str2, "%Y-%m-%d %H:%M:%S")

# 计算时间差
time_difference = time2 - time1

# 提取时间差中的小时部分
hours_difference = time_difference.total_seconds() / 3600

print(f"时间差为 {hours_difference} 小时")