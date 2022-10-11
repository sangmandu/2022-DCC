# json 파일 설명
- 전체_실사_이름모음(json).json
  - key : all ( 1개 )
- 클래스_실사_이름모음(json).json
  - key : 'L2_24', 'L2_41', 'L2_50', 'L2_12', 'L2_3' ( 5개 )
  
# json 파일 불러오기

```
# 불러오기(json)
import json
json_all = open('경로/전체_실사_이름모음(json).json', encoding = 'utf-8')
json_class = open('경로/클래스_실사_이름모음(json).json', encoding = 'utf-8')

dict_all = json.load(json_all)	# type : dict
dict_class = json.load(json_class)  # type : dict
```

```
dict_all
>>> {'all': ['qrlfbvbyjspvknbdkywt.jpg',
  'xufwaryzxpnpafvsjgqc.jpg',
  'egrbjoxbrdpdxuumhnao.jpg',
  'jbznycnnmmtjrcsaaucs.jpg',
  'pglqgsthdgtegfacdvjt.jpg',
  'bbuglcjnjcnmrpfxznjh.jpg',
  'dzaonnhttixxekrhaitq.jpg',
  'xyglhigdjaeclqyhljpk.jpg',
  'mljnxutyslelihenedde.jpg',
  ...
```
