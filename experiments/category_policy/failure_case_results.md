# Failure Case Evaluation Report (N=70 Baseline Failures)

## Summary
- **Baseline Failures Evaluated**: 70
- **Corrections (Failures Fixed)**: **26** (37.14% correction rate)
- **Harms (Previously Correct Turned Incorrect)**: **6** (3.45% harm rate)
- **Net Improvement**: **+20** screenshots
- **Baseline Top-1 Accuracy**: 72.54% (177/244)
- **Policy Top-1 Accuracy**: 80.74% (197/244)

## Per-Screenshot Failure Resolution Table
| ID | Filename | Ground Truth | Baseline Pred | Policy Pred | Status | Policy Reasons |
|---|---|---|---|---|---|---|
| ext2_013 | ext2_013.jpg | General | Work & Career | Work & Career | **UNCHANGED** | General vs Work & Career: Workplace context tiebreaker applied to Work & Career |
| ext2_014 | ext2_014.jpg | General | Work & Career | Work & Career | **UNCHANGED** | General vs Work & Career: Workplace context tiebreaker applied to Work & Career |
| ext2_021 | ext2_021.jpg | General | Work & Career | Work & Career | **UNCHANGED** | General vs Work & Career: Workplace context tiebreaker applied to Work & Career; General vs Shopping: Retail item context tiebreaker applied to Shopping |
| ext2_023 | ext2_023.jpg | Work & Career | General | Work & Career | **CORRECTED** | General vs Work & Career: Workplace context tiebreaker applied to Work & Career |
| ext2_024 | ext2_024.jpg | Education | Shopping | Education | **CORRECTED** | Academic lecture / Course syllabus primary object; General vs Communication: Messaging context tiebreaker applied to Communication; General vs Shopping: Retail item context tiebreaker applied to Shopping |
| ext2_027 | ext2_027.jpg | General | Education | Education | **UNCHANGED** | General vs Communication: Messaging context tiebreaker applied to Communication |
| ext2_032 | ext2_032.jpg | Shopping | Entertainment | Shopping | **CORRECTED** | None |
| ext2_034 | ext2_034.jpg | General | Shopping | Shopping | **UNCHANGED** | General vs Shopping: Retail item context tiebreaker applied to Shopping |
| ext2_042 | ext2_042.jpg | General | Shopping | Shopping | **UNCHANGED** | None |
| ext2_046 | ext2_046.jpg | Shopping | General | Shopping | **CORRECTED** | General vs Shopping: Retail item context tiebreaker applied to Shopping |
| ext2_047 | ext2_047.jpg | Shopping | General | Shopping | **CORRECTED** | General vs Shopping: Retail item context tiebreaker applied to Shopping |
| ext2_048 | ext2_048.jpg | Shopping | General | Shopping | **CORRECTED** | General vs Shopping: Retail item context tiebreaker applied to Shopping |
| ext2_064 | ext2_064.jpg | General | Shopping | Shopping | **UNCHANGED** | None |
| ext2_068 | ext2_068.jpg | Shopping | Travel | Shopping | **CORRECTED** | General vs Shopping: Retail item context tiebreaker applied to Shopping |
| ext2_069 | ext2_069.jpg | Shopping | Travel | Shopping | **CORRECTED** | General vs Shopping: Retail item context tiebreaker applied to Shopping |
| ext2_070 | ext2_070.jpg | Shopping | Travel | Shopping | **CORRECTED** | None |
| ext2_071 | ext2_071.jpg | Shopping | Travel | Shopping | **CORRECTED** | General vs Shopping: Retail item context tiebreaker applied to Shopping |
| ext2_079 | ext2_079.jpg | Connectivity | General | Connectivity | **CORRECTED** | General vs Work & Career: Workplace context tiebreaker applied to Work & Career |
| ext2_084 | ext2_084.jpg | General | Education | Education | **UNCHANGED** | None |
| ext2_086 | ext2_086.jpg | Work & Career | Entertainment | Entertainment | **UNCHANGED** | None |
| ext2_087 | ext2_087.jpg | Work & Career | General | General | **UNCHANGED** | None |
| ext2_092 | ext2_092.jpg | General | Payment | Payment | **UNCHANGED** | General vs Communication: Messaging context tiebreaker applied to Communication |
| ext2_093 | ext2_093.jpg | Education | Payment | Education | **CORRECTED** | Academic lecture / Course syllabus primary object |
| ext2_094 | ext2_094.jpg | General | Payment | Payment | **UNCHANGED** | None |
| ext2_096 | ext2_096.jpg | General | Payment | Payment | **UNCHANGED** | None |
| ext2_098 | ext2_098.jpg | General | Payment | Payment | **UNCHANGED** | None |
| ext2_105 | ext2_105.jpg | General | Payment | Payment | **UNCHANGED** | None |
| ext2_106 | ext2_106.jpg | General | Payment | Payment | **UNCHANGED** | None |
| ext2_107 | ext2_107.jpg | General | Payment | Payment | **UNCHANGED** | None |
| ext2_108 | ext2_108.jpg | General | Payment | Payment | **UNCHANGED** | None |
| ext2_110 | ext2_110.jpg | Payment | Communication | Communication | **UNCHANGED** | General vs Communication: Messaging context tiebreaker applied to Communication |
| ext2_111 | ext2_111.jpg | Education | Work & Career | Education | **CORRECTED** | Academic lecture / Course syllabus primary object; General vs Work & Career: Workplace context tiebreaker applied to Work & Career |
| ext2_113 | ext2_113.jpg | Education | General | Education | **CORRECTED** | Academic lecture / Course syllabus primary object |
| ext2_114 | ext2_114.jpg | General | Work & Career | Work & Career | **UNCHANGED** | None |
| ext2_122 | ext2_122.jpg | Payment | Communication | Communication | **UNCHANGED** | Email/Chat correspondence layout primary object; General vs Work & Career: Workplace context tiebreaker applied to Work & Career; General vs Communication: Messaging context tiebreaker applied to Communication |
| ext2_126 | ext2_126.jpg | Education | Work & Career | Education | **CORRECTED** | Academic lecture / Course syllabus primary object; General vs Work & Career: Workplace context tiebreaker applied to Work & Career |
| ext2_127 | ext2_127.jpg | Education | Work & Career | Education | **CORRECTED** | Academic lecture / Course syllabus primary object; General vs Work & Career: Workplace context tiebreaker applied to Work & Career; General vs Communication: Messaging context tiebreaker applied to Communication |
| ext2_128 | ext2_128.jpg | Education | Work & Career | Education | **CORRECTED** | Academic lecture / Course syllabus primary object; General vs Work & Career: Workplace context tiebreaker applied to Work & Career |
| ext2_130 | ext2_130.jpg | Communication | Work & Career | Work & Career | **UNCHANGED** | Email/Chat correspondence layout primary object; General vs Work & Career: Workplace context tiebreaker applied to Work & Career; General vs Communication: Messaging context tiebreaker applied to Communication |
| ext2_133 | ext2_133.jpg | Communication | Work & Career | Communication | **CORRECTED** | Email/Chat correspondence layout primary object; General vs Communication: Messaging context tiebreaker applied to Communication |
| ext2_138 | ext2_138.jpg | Education | Work & Career | Education | **CORRECTED** | Academic lecture / Course syllabus primary object; General vs Work & Career: Workplace context tiebreaker applied to Work & Career |
| ext2_139 | ext2_139.jpg | General | Travel | Travel | **UNCHANGED** | None |
| ext2_143 | ext2_143.jpg | Education | General | Education | **CORRECTED** | Academic lecture / Course syllabus primary object |
| ext2_147 | ext2_147.jpg | Entertainment | General | Entertainment | **CORRECTED** | None |
| ext2_150 | ext2_150.jpg | Communication | Work & Career | Communication | **CORRECTED** | Email/Chat correspondence layout primary object; General vs Work & Career: Workplace context tiebreaker applied to Work & Career; General vs Communication: Messaging context tiebreaker applied to Communication |
| ext2_153 | ext2_153.jpg | Payment | Travel | Payment | **CORRECTED** | None |
| ext2_158 | ext2_158.jpg | Communication | Work & Career | Work & Career | **UNCHANGED** | Email/Chat correspondence layout primary object; Job portal / Career vacancy layout primary object; General vs Communication: Messaging context tiebreaker applied to Communication |
| ext2_160 | ext2_160.jpg | General | Entertainment | Entertainment | **UNCHANGED** | None |
| ext2_162 | ext2_162.jpg | General | Travel | Travel | **UNCHANGED** | None |
| ext2_163 | ext2_163.jpg | Work & Career | Payment | Work & Career | **CORRECTED** | General vs Work & Career: Workplace context tiebreaker applied to Work & Career |
| ext2_180 | ext2_180.jpg | General | Entertainment | Entertainment | **UNCHANGED** | None |
| ext2_193 | ext2_193.jpg | General | Entertainment | Entertainment | **UNCHANGED** | None |
| ext2_200 | ext2_200.jpg | General | Entertainment | Entertainment | **UNCHANGED** | None |
| ext2_201 | ext2_201.jpg | General | Entertainment | Entertainment | **UNCHANGED** | General vs Work & Career: Workplace context tiebreaker applied to Work & Career |
| ext2_202 | ext2_202.jpg | Shopping | General | General | **UNCHANGED** | None |
| ext2_206 | ext2_206.jpg | Education | Shopping | Shopping | **UNCHANGED** | None |
| ext2_207 | ext2_207.jpg | General | Work & Career | Work & Career | **UNCHANGED** | General vs Work & Career: Workplace context tiebreaker applied to Work & Career; General vs Communication: Messaging context tiebreaker applied to Communication |
| ext2_210 | ext2_210.jpg | General | Education | Education | **UNCHANGED** | None |
| ext2_211 | ext2_211.jpg | Education | Shopping | Shopping | **UNCHANGED** | IDE editor / Terminal traceback primary object; General vs Shopping: Retail item context tiebreaker applied to Shopping |
| ext2_212 | ext2_212.jpg | General | Communication | Entertainment | **UNCHANGED** | None |
| ext2_213 | ext2_213.jpg | General | Communication | Communication | **UNCHANGED** | None |
| ext2_217 | ext2_217.jpg | General | Travel | Travel | **UNCHANGED** | None |
| ext2_224 | ext2_224.jpg | General | Communication | Entertainment | **UNCHANGED** | None |
| ext2_227 | ext2_227.jpg | General | Entertainment | Entertainment | **UNCHANGED** | None |
| ext2_228 | ext2_228.jpg | Communication | Entertainment | Communication | **CORRECTED** | General vs Communication: Messaging context tiebreaker applied to Communication |
| ext2_231 | ext2_231.jpg | Shopping | Work & Career | Work & Career | **UNCHANGED** | General vs Work & Career: Workplace context tiebreaker applied to Work & Career |
| ext2_232 | ext2_232.jpg | Education | Work & Career | Education | **CORRECTED** | Academic lecture / Course syllabus primary object; General vs Communication: Messaging context tiebreaker applied to Communication |
| ext2_233 | ext2_233.jpg | General | Work & Career | Work & Career | **UNCHANGED** | General vs Work & Career: Workplace context tiebreaker applied to Work & Career; General vs Communication: Messaging context tiebreaker applied to Communication |
| ext2_237 | ext2_237.jpg | Shopping | Communication | Communication | **UNCHANGED** | General vs Communication: Messaging context tiebreaker applied to Communication |
| ext2_240 | ext2_240.jpg | General | Payment | Payment | **UNCHANGED** | None |