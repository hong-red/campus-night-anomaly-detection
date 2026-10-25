import zipfile, re
z = zipfile.ZipFile("cross_validate_schools_3beh.xlsx")
x = z.read("xl/charts/chart1.xml").decode("utf-8")
print("len", len(x), "series", x.count("<c:ser>"))
for m in re.finditer(r"<c:f>(.*?)</c:f>", x):
    print("ref:", m.group(1))
for m in re.finditer(r'srgbClr val="(.*?)"', x):
    print("color:", m.group(1))
print("has barChart:", "<c:barChart>" in x, "barDir:", re.search(r'barDir val="(\w+)"', x).group(1))
