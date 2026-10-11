# Rule Impact Analysis Report

## Summary of Triggered Rule Predictions on External V2
- **Total Prediction Changes**: 34
- **Positive Corrections**: 26
- **Harms Introduced**: 6

## Detailed Prediction Impact Table
| ID | Filename | Ground Truth | Baseline Pred | New Policy Pred | Impact Status | Policy Reasons |
|---|---|---|---|---|---|---|
| ext2_023 | ext2_023.jpg | Work & Career | General | Work & Career | **CORRECTED** | General vs Work & Career: Workplace context tiebreaker applied to Work & Career |
| ext2_024 | ext2_024.jpg | Education | Shopping | Education | **CORRECTED** | Academic lecture / Course syllabus primary object; General vs Communication: Messaging context tiebreaker applied to Communication; General vs Shopping: Retail item context tiebreaker applied to Shopping |
| ext2_032 | ext2_032.jpg | Shopping | Entertainment | Shopping | **CORRECTED** | None |
| ext2_037 | ext2_037.jpg | General | General | Shopping | **HARMED** | E-commerce product catalog / cart primary object; General vs Communication: Messaging context tiebreaker applied to Communication; General vs Shopping: Retail item context tiebreaker applied to Shopping |
| ext2_038 | ext2_038.jpg | General | General | Shopping | **HARMED** | None |
| ext2_046 | ext2_046.jpg | Shopping | General | Shopping | **CORRECTED** | General vs Shopping: Retail item context tiebreaker applied to Shopping |
| ext2_047 | ext2_047.jpg | Shopping | General | Shopping | **CORRECTED** | General vs Shopping: Retail item context tiebreaker applied to Shopping |
| ext2_048 | ext2_048.jpg | Shopping | General | Shopping | **CORRECTED** | General vs Shopping: Retail item context tiebreaker applied to Shopping |
| ext2_058 | ext2_058.jpg | General | General | Shopping | **HARMED** | General vs Shopping: Retail item context tiebreaker applied to Shopping |
| ext2_066 | ext2_066.jpg | General | General | Shopping | **HARMED** | None |
| ext2_068 | ext2_068.jpg | Shopping | Travel | Shopping | **CORRECTED** | General vs Shopping: Retail item context tiebreaker applied to Shopping |
| ext2_069 | ext2_069.jpg | Shopping | Travel | Shopping | **CORRECTED** | General vs Shopping: Retail item context tiebreaker applied to Shopping |
| ext2_070 | ext2_070.jpg | Shopping | Travel | Shopping | **CORRECTED** | None |
| ext2_071 | ext2_071.jpg | Shopping | Travel | Shopping | **CORRECTED** | General vs Shopping: Retail item context tiebreaker applied to Shopping |
| ext2_079 | ext2_079.jpg | Connectivity | General | Connectivity | **CORRECTED** | General vs Work & Career: Workplace context tiebreaker applied to Work & Career |
| ext2_093 | ext2_093.jpg | Education | Payment | Education | **CORRECTED** | Academic lecture / Course syllabus primary object |
| ext2_111 | ext2_111.jpg | Education | Work & Career | Education | **CORRECTED** | Academic lecture / Course syllabus primary object; General vs Work & Career: Workplace context tiebreaker applied to Work & Career |
| ext2_113 | ext2_113.jpg | Education | General | Education | **CORRECTED** | Academic lecture / Course syllabus primary object |
| ext2_126 | ext2_126.jpg | Education | Work & Career | Education | **CORRECTED** | Academic lecture / Course syllabus primary object; General vs Work & Career: Workplace context tiebreaker applied to Work & Career |
| ext2_127 | ext2_127.jpg | Education | Work & Career | Education | **CORRECTED** | Academic lecture / Course syllabus primary object; General vs Work & Career: Workplace context tiebreaker applied to Work & Career; General vs Communication: Messaging context tiebreaker applied to Communication |
| ext2_128 | ext2_128.jpg | Education | Work & Career | Education | **CORRECTED** | Academic lecture / Course syllabus primary object; General vs Work & Career: Workplace context tiebreaker applied to Work & Career |
| ext2_133 | ext2_133.jpg | Communication | Work & Career | Communication | **CORRECTED** | Email/Chat correspondence layout primary object; General vs Communication: Messaging context tiebreaker applied to Communication |
| ext2_138 | ext2_138.jpg | Education | Work & Career | Education | **CORRECTED** | Academic lecture / Course syllabus primary object; General vs Work & Career: Workplace context tiebreaker applied to Work & Career |
| ext2_142 | ext2_142.jpg | General | General | Work & Career | **HARMED** | General vs Work & Career: Workplace context tiebreaker applied to Work & Career |
| ext2_143 | ext2_143.jpg | Education | General | Education | **CORRECTED** | Academic lecture / Course syllabus primary object |
| ext2_146 | ext2_146.jpg | General | General | Communication | **HARMED** | General vs Communication: Messaging context tiebreaker applied to Communication |
| ext2_147 | ext2_147.jpg | Entertainment | General | Entertainment | **CORRECTED** | None |
| ext2_150 | ext2_150.jpg | Communication | Work & Career | Communication | **CORRECTED** | Email/Chat correspondence layout primary object; General vs Work & Career: Workplace context tiebreaker applied to Work & Career; General vs Communication: Messaging context tiebreaker applied to Communication |
| ext2_153 | ext2_153.jpg | Payment | Travel | Payment | **CORRECTED** | None |
| ext2_163 | ext2_163.jpg | Work & Career | Payment | Work & Career | **CORRECTED** | General vs Work & Career: Workplace context tiebreaker applied to Work & Career |
| ext2_212 | ext2_212.jpg | General | Communication | Entertainment | **OTHER_CHANGE** | None |
| ext2_224 | ext2_224.jpg | General | Communication | Entertainment | **OTHER_CHANGE** | None |
| ext2_228 | ext2_228.jpg | Communication | Entertainment | Communication | **CORRECTED** | General vs Communication: Messaging context tiebreaker applied to Communication |
| ext2_232 | ext2_232.jpg | Education | Work & Career | Education | **CORRECTED** | Academic lecture / Course syllabus primary object; General vs Communication: Messaging context tiebreaker applied to Communication |