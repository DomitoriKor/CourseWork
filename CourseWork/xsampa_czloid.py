import os
from openpyxl import load_workbook

import json

#Build CMU - X-SAMPA dictionary
#cmu_xsampa_dict = {"ax": "V", "ey": "e", "aa": "a", "ae": "{", "ah": "@", "ao": "O",
#            "aw": "aU", "ay": "aI", "ch": "tS", "dh": "D", "eh": "e", "er": "@r",
#            "hh": "h", "ih": "I", "jh": "dZ", "ng": "N",  "ow": "oU", "oy": "OI",
#            "sh": "S", "th": "T", "uh": "U", "uw": "u", "zh": "Z", "iy": "i", "y": "j",
#            "dx": "4", "ix": "1", "ux": "}", "el": "@l", "em" : "@m",
#            "en": "@n", "nx": "r~", "q": "?", "wh": "W"}

#cmu_xsampa_dict = {"ax": "V", "ey": "e", "aa": "a", "ae": "{", "ah": "@", "ao": "O",
#            "aw": "aU", "ay": "aI", "ch": "tS", "dh": "D", "eh": "E", "er": "@r",
#            "hh": "h", "ih": "I", "jh": "dZ", "ng": "N",  "ow": "oU", "oy": "OI",
#            "sh": "S", "th": "T", "uh": "U", "uw": "u", "zh": "Z", "iy": "i", "y": "j",
#            "dx": "4", "ix": "1", "ux": "}", "b": "b", "d": "d", "el": "@l", "em" : "@m",
#            "en": "@n", "f": "f", "g": "g", "k": "k", "l": "l", "m": "m", "n": "n",
#            "nx": "r~", "p": "p", "q": "?", "r": "r", "s": "s", "t": "t", "v": "v",
#            "w": "w", "wh": "W", "z": "z"}

cmu_xsampa_dict = {"ax": "@", "ey": "eI", "aa": "a", "ae": "{", "ah": "V", "ao": "O",
            "aw": "aU", "ay": "aI", "ch": "tS", "dh": "D", "eh": "e", "er": "@r",
            "hh": "h", "ih": "I", "jh": "dZ", "ng": "N",  "ow": "oU", "oy": "OI",
            "sh": "S", "th": "T", "uh": "U", "uw": "u", "zh": "Z", "iy": "i", "y": "j",
            "dx": "4", "ix": "1", "ux": "}", "el": "@l", "em" : "@m",
            "en": "@n", "nx": "r~", "q": "?", "wh": "W"}

print(cmu_xsampa_dict)

with open(os.path.join(r"D:\Projects\CourseWork\CourseWork\resources", "cmu_xsampa.dict"), "w") as cmu_xsampa_dict_file:
    json.dump(cmu_xsampa_dict, cmu_xsampa_dict_file)


#Build X-SAMPA - CZloid dictionary
wb = load_workbook('D:\VCCV.xlsx')
trans_table = wb.active

xsampa_cz_table = list()

#Get vowels list
vowel_set = set()
for vowel_line in trans_table['A3':'B41']:
    cz = vowel_line[0].value
    xsampa = vowel_line[1].value
    if isinstance(cz, (int)):
        cz = str(cz)
    if isinstance(xsampa, (int)):
        xsampa = str(xsampa)
    vowel_set.add(cz)
    xsampa_cz_table.append((xsampa, cz))

#Get CV consonants list
consonant_set = set()
for cv_cons_line in trans_table['A86':'B123']:
    cz = cv_cons_line[0].value
    xsampa = cv_cons_line[1].value
    if isinstance(cz, (int)):
        cz = str(cz)
    if isinstance(xsampa, (int)):
        xsampa = str(xsampa)
    consonant_set.add(cz)
    xsampa_cz_table.append((xsampa, cz))

xsampa_cz_dict = dict(xsampa_cz_table)

print(xsampa_cz_dict)

with open(os.path.join(r"D:\Projects\CourseWork\CourseWork\resources", "xsampa_cz.dict"), "w") as xsampa_cz_dict_file:
    json.dump(xsampa_cz_dict, xsampa_cz_dict_file)

#Build CMU - CZloid dictionary
cmu_cz_table = list()
for cmu in cmu_xsampa_dict.keys():
    xsampa = cmu_xsampa_dict[cmu]
    if xsampa in xsampa_cz_dict.keys():
        cz = xsampa_cz_dict[xsampa]
        cmu_cz_table.append((cmu, cz))

cmu_cz_dict = dict(cmu_cz_table)

print(cmu_cz_dict)

with open(os.path.join(r"D:\Projects\CourseWork\CourseWork\resources", "cmu_cz.dict"), "w") as cmu_cz_dict_file:
    json.dump(cmu_cz_dict, cmu_cz_dict_file)

cmu_ipa_dict = {"ax": "ʌ", "ey": "eɪ", "aa": "ɑ", "ae": "æ", "ah": "ə", "ao": "ɔ",
                "aw": "aʊ", "ay": "aɪ", "ch": "tʃ", "dh": "ð", "eh": "ɛ", "er": "ər",
                "hh": "h", "ih": "ɪ", "jh": "dʒ", "ng": "ŋ",  "ow": "oʊ", "oy": "ɔɪ",
                "sh": "ʃ", "th": "θ", "uh": "ʊ", "uw": "u", "zh": "ʒ", "iy": "i", "y": "j",
                "dx": "ɾ", "ix": "ɨ", "ux": "ʉ", "el": "əl", "em" : "əm",
                "en": "ən", "nx": "ɾ̃", "q": "ʔ", "wh": "ʍ"}

with open(os.path.join(r"D:\Projects\CourseWork\CourseWork\resources", "cmu_ipa.dict"), "w") as cmu_ipa_dict_file:
    json.dump(cmu_ipa_dict, cmu_ipa_dict_file)

for c in cmu_cz_dict.keys():
    if not c in cmu_ipa_dict.keys():
        print(c)