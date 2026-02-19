#!/usr/bin/python3
# Copyright (c) 2021-2024, SIL Global.
# Licensed under MIT license: https://opensource.org/licenses/MIT

import enum
import re


class Cats(enum.Enum):
  Other = 0
  Base = 1
  Robat = 2
  Coeng = 3
  Shift = 4
  Z = 5
  VPre = 6
  VB = 7
  VA = 8
  VPost = 9
  MS = 10
  MF = 11
  ZFCoeng = 12


categories = (
  [Cats.Base] * 35  # 1780-17A2
  + [Cats.Other] * 2  # 17A3-17A4
  + [Cats.Base] * 15  # 17A5-17B3
  + [Cats.Other] * 2  # 17B4-17B5
  + [Cats.VPost]  # 17B6
  + [Cats.VA] * 4  # 17B7-17BA
  + [Cats.VB] * 3  # 17BB-17BD
  + [Cats.VPre] * 8  # 17BE-17C5
  + [Cats.MS]  # 17C6
  + [Cats.MF] * 2  # 17C7-17C8
  + [Cats.Shift] * 2  # 17C9-17CA
  + [Cats.MS]  # 17CB
  + [Cats.Robat]  # 17CC
  + [Cats.MS] * 5  # 17CD-17D1
  + [Cats.Coeng]  # 17D2
  + [Cats.MS]  # 17D3
  + [Cats.Other] * 9  # 17D4-17DC
  + [Cats.MS]
)  # 17DD

khres = {  # useful regular sub expressions used later
  # All bases
  "B": "[\u1780-\u17a2\u17a5-\u17b3\u25cc]",
  # All consonants excluding Ro
  "NonRo": "[\u1780-\u1799\u179b-\u17a2\u17a5-\u17b3]",
  # All consonants exclude Bo
  "NonBA": "[\u1780-\u1793\u1795-\u17a2\u17a5-\u17b3]",
  # Series 1 consonants
  "S1": "[\u1780-\u1783\u1785-\u1788\u178a-\u178d\u178f-\u1792"
  "\u1795-\u1797\u179e-\u17a0\u17a2]",
  # Series 2 consonants
  "S2": "[\u1784\u1780\u178e\u1793\u1794\u1798-\u179d\u17a1\u17a3-\u17b3]",
  # Simple following Vowel in Modern Khmer
  "VA": "(?:[\u17b7-\u17ba\u17be\u17bf\u17dd]|\u17b6\u17c6)",
  # Above vowel (as per shifter rules) with vowel sequences
  "VAX": "(?:[\u17c1-\u17c5]?{VA})",
  # Above vowel with samyok (modern khmer)
  "VAS": "(?:{VA}|[\u17c1-\u17c3]?\u17d0)",
  # Above vowel with samyok (middle khmer)
  "VASX": "(?:{VAX}|[\u17c1-\u17c3]?\u17d0)",
  # Below vowel (with Middle Khmer prefix)
  "VB": "(?:[\u17c1-\u17c3]?[\u17bb-\u17bd])",
  # contains series 1 and no BA
  "STRONG": """  {S1}\u17cc?                 # series 1 robat?
                    (?:\u17d2{NonBA}            # nonba coengs
                       (?:\u17d2{NonBA})?)?
                  | {NonBA}\u17cc?              # nonba robat?
                    (?:  \u17d2{S1}               # series 1 coeng
                         (?:\u17d2{NonBA})?       #   + any nonba coeng
                       | \u17d2{NonBA}\u17d2{S1}  # nonba coeng + series 1 coeng
                    )""",
  # contains BA or only series 2
  "NSTRONG": """(?:{S2}\u17cc?(?:\u17d2{S2}(?:\u17d2{S2})?)? # Series 2 + series 2 coengs
                     |\u1794\u17cc?(?:{COENG}(?:{COENG})?)?    # or ba with any coeng
                     |{B}\u17cc?(?:\u17d2{NonRo}\u17d2\u1794   # or ba coeng
                                  |\u17d2\u1794(?:\u17d2{B})))""",
  "COENG": "(?:(?:\u17d2{NonRo})?\u17d2{B})",
  # final coeng
  "FCOENG": "(?:\u200d(?:\u17d2{NonRo})+)",
  # Allowed shifter sequences in Modern Khmer
  "SHIFT": """(?:  (?<={STRONG}) \u17ca\u200c (?={VA})     # strong + triisap held up
                     | (?<={NSTRONG})\u17c9\u200c (?={VAS})    # weak + muusikatoan held up
                     | [\u17c9\u17ca]                          # any shifter
                  )""",
  # Allowed shifter sequences in Middle Khmer
  "SHIFTX": """(?:(?<={STRONG}) \u17ca\u200c (?={VAX})      # strong + triisap held up
                    | (?<={NSTRONG})\u17c9\u200c (?={VASX})    # weak + muusikatoan held up
                    | [\u17c9\u17ca]                           # any shifter
                  )""",
  # Modern Khmer vowel
  "V": "[\u17b6-\u17c5]?",
  # Middle Khmer vowel sequences (not worth trying to unpack this)
  "VX": "(?:\u17c1[\u17bc\u17bd]?[\u17b7\u17b9\u17ba]?|"
  "[\u17c2\u17c3]?[\u17bc\u17bd]?[\u17b7-\u17ba]\u17b6|"
  "[\u17c2\u17c3]?[\u17bb-\u17bd]?\u17b6|\u17be[\u17bc\u17bd]?\u17b6?|"
  "[\u17c1-\u17c5]?\u17bb(?![\u17d0\u17dd])|"
  "[\u17bf\u17c0]|[\u17c2-\u17c5]?[\u17bc\u17bd]?[\u17b7-\u17ba]?)",
  # Modern Khmer Modifiers
  "MS": """(?:(?:  [\u17c6\u17cb\u17cd-\u17cf\u17d1\u17d3]   # follows anything
                       | (?<!\u17bb) [\u17d0\u17dd])                # not after -u
                     [\u17c6\u17cb\u17cd-\u17d1\u17d3\u17dd]?  # And an optional second
                  )""",
  # Middle Khmer Modifiers
  "MSX": """(?:(?:  [\u17c6\u17cb\u17cd-\u17cf\u17d1\u17d3]   # follows anything
                        | (?<!\u17bb [\u17b6\u17c4\u17c5]?)       # blocking -u sequence
                        [\u17d0\u17dd])                           # for these modifiers
                     [\u17c6\u17cb\u17cd-\u17d1\u17d3\u17dd]? # And an optional second
                  )""",
}

# expand 3 times: SHIFTX -> VASX -> VAX -> VA
for i in range(3):
  khres = {k: v.format(**khres) for k, v in khres.items()}


def charcat(c):
  """Returns the Khmer character category for a single char string"""
  o = ord(c)
  if 0x1780 <= o <= 0x17DD:
    return categories[o - 0x1780]
  elif o == 0x200C:
    return Cats.Z
  elif o == 0x200D:
    return Cats.ZFCoeng
  return Cats.Other


def lunar(m, base):
  """Returns the lunar date symbol from the appropriate set base"""
  v = (ord(m.group(1) or "\u17e0") - 0x17E0) * 10 + ord(m.group(2)) - 0x17E0
  if v > 15:  # translate \u17D4\u17D2\u17E0 as well
    return m.group(0)
  return chr(v + base)


def khnormal(txt, lang="km"):
  """Returns khmer normalised string, without fixing or marking errors"""
  # Mark final coengs in Middle Khmer
  if lang == "xhm":
    txt = re.sub(r"([\u17B6-\u17C5]\u17D2)", "\u200d\\1", txt)
  # Categorise every character in the string
  charcats = [charcat(c) for c in txt]

  # Recategorise base -> coeng after coeng char (or ZFCoeng)
  for i in range(1, len(charcats)):
    if txt[i - 1] in "\u200d\u17d2" and charcats[i] in (Cats.Base, Cats.Coeng):
      charcats[i] = charcats[i - 1]

  # Find subranges of base+non other and sort components in the subrange
  i = 0
  res = []
  while i < len(charcats):
    c = charcats[i]
    if c != Cats.Base:
      res.append(txt[i])
      i += 1
      continue
    # Scan for end of syllable
    j = i + 1
    while j < len(charcats) and charcats[j].value > Cats.Base.value:
      j += 1
    # Sort syllable based on character categories
    # Sort the char indices by category then position in string
    newindices = sorted(range(i, j), key=lambda e: (charcats[e].value, e))
    replaces = "".join(txt[n] for n in newindices)

    replaces = re.sub(
      "(\u200d?\u17d2)[\u17d2\u200c\u200d]+", r"\1", replaces
    )  # remove multiple invisible chars
    replaces = re.sub("\u17be\u17b6", "\u17c4\u17b8", replaces)  # confusable vowels
    # map compoound vowel sequences to compounds with -u before to be converted
    replaces = re.sub("\u17c1([\u17bb-\u17bd]?)\u17b8", "\u17be\\1", replaces)
    replaces = re.sub("\u17c1([\u17bb-\u17bd]?)\u17b6", "\u17c4\\1", replaces)
    replaces = re.sub("(\u17be)(\u17bb)", r"\2\1", replaces)
    # Replace -u + upper vowel with consonant shifter
    replaces = re.sub(
      ("((?:{STRONG})[\u17c1-\u17c5]?)\u17bb" + "(?={VA}|\u17d0)").format(**khres),
      "\\1\u17ca",
      replaces,
      flags=re.X,
    )
    replaces = re.sub(
      ("((?:{NSTRONG})[\u17c1-\u17c5]?)\u17bb" + "(?={VA}|\u17d0)").format(**khres),
      "\\1\u17c9",
      replaces,
      flags=re.X,
    )
    replaces = re.sub(
      "(\u17d2\u179a)(\u17d2[\u1780-\u17b3])", r"\2\1", replaces
    )  # coeng ro second
    # replaces = re.sub("(\u17d2)\u178a", "\\1\u178f", replaces)  # coeng da->ta
    # convert lunar dates from old style to use lunar date symbols
    replaces = re.sub(
      "(\u17e1?)([\u17e0-\u17e9])\u17d2\u17d4", lambda m: lunar(m, 0x19E0), replaces
    )
    replaces = re.sub(
      "\u17d4\u17d2(\u17e1?)([\u17e0-\u17e9])", lambda m: lunar(m, 0x19F0), replaces
    )
    replaces = re.sub("\u17d4\u17d2\u17d4", "\u19f0", replaces)
    res.append(replaces)
    i = j
  return "".join(res)
