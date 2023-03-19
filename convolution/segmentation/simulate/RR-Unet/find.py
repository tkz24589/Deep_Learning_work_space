result = []
# 打开列表文件
with open("up(1).list", "r") as file, open("output.txt", "w") as output_file:
    # 遍历文件中的每一行文本
    for line in file:
        # print(line)
        # 如果行中包含字符串 "CPUN"，则打印该行文本
        if "CPUN" in line:
            line_list = line.split("CPUN")
            for i in range(1, len(line_list)):
                output_file.write(line_list[i].split('|')[1]+ '\n')