# External V2 Failure Correction Analysis Report

## Executive Summary
- **Total Screenshots**: 244
- **Baseline Failures Evaluated**: 70
- **Baseline Top-1 Accuracy**: 72.54% (177/244)
- **Final Visual Signal Top-1 Accuracy**: 72.54% (177/244)
- **Corrections (Failures Fixed)**: **0** (0.00% correction rate)
- **Harms (Previously Correct Turned Incorrect)**: **0** (0.00% harm rate)
- **Net Improvement**: **+0** screenshots
- **Unchanged Failures**: 70

## Detailed Resolution Table of Baseline Failures
| ID | Filename | Ground Truth | Baseline Pred | Visual Type | Visual Conf | Final Pred | Status | Reranking Reason |
|---|---|---|---|---|---|---|---|---|
| ext2_013 | ext2_013.jpg | General | Work & Career | dashboard_settings | 0.2937 | Work & Career | **UNCHANGED** | Base Top-1 'Work & Career' retained (score=0.5793) |
| ext2_014 | ext2_014.jpg | General | Work & Career | document_article_view | 0.2673 | Work & Career | **UNCHANGED** | Base Top-1 'Work & Career' retained (score=0.4090) |
| ext2_021 | ext2_021.jpg | General | Work & Career | travel_booking | 0.3202 | Work & Career | **UNCHANGED** | Base Top-1 'Work & Career' retained (score=0.5900) |
| ext2_023 | ext2_023.jpg | Work & Career | General | travel_booking | 0.2862 | General | **UNCHANGED** | Base Top-1 'General' retained via High-Confidence Guardrail |
| ext2_024 | ext2_024.jpg | Education | Shopping | travel_booking | 0.3286 | Shopping | **UNCHANGED** | Base Top-1 'Shopping' retained (score=0.4895) |
| ext2_027 | ext2_027.jpg | General | Education | code_editor_terminal | 0.2579 | Education | **UNCHANGED** | Base Top-1 'Education' retained (score=0.5928) |
| ext2_032 | ext2_032.jpg | Shopping | Entertainment | document_article_view | 0.2842 | Entertainment | **UNCHANGED** | Base Top-1 'Entertainment' retained (score=0.3580) |
| ext2_034 | ext2_034.jpg | General | Shopping | travel_booking | 0.2788 | Shopping | **UNCHANGED** | Base Top-1 'Shopping' retained (score=0.3762) |
| ext2_042 | ext2_042.jpg | General | Shopping | travel_booking | 0.3420 | Shopping | **UNCHANGED** | Promoted 'Shopping' (score=0.3417) over base Top-1 'General' via Multi-Signal Evidence |
| ext2_046 | ext2_046.jpg | Shopping | General | travel_booking | 0.3540 | General | **UNCHANGED** | Base Top-1 'General' retained (score=0.2672) |
| ext2_047 | ext2_047.jpg | Shopping | General | document_article_view | 0.2446 | General | **UNCHANGED** | Base Top-1 'General' retained (score=0.2696) |
| ext2_048 | ext2_048.jpg | Shopping | General | travel_booking | 0.3605 | General | **UNCHANGED** | Base Top-1 'General' retained (score=0.2711) |
| ext2_064 | ext2_064.jpg | General | Shopping | travel_booking | 0.2718 | Shopping | **UNCHANGED** | Base Top-1 'Shopping' retained (score=0.5396) |
| ext2_068 | ext2_068.jpg | Shopping | Travel | travel_booking | 0.3776 | Travel | **UNCHANGED** | Base Top-1 'Travel' retained (score=0.3326) |
| ext2_069 | ext2_069.jpg | Shopping | Travel | travel_booking | 0.4008 | Travel | **UNCHANGED** | Base Top-1 'Travel' retained (score=0.3176) |
| ext2_070 | ext2_070.jpg | Shopping | Travel | travel_booking | 0.3970 | Travel | **UNCHANGED** | Base Top-1 'Travel' retained (score=0.3989) |
| ext2_071 | ext2_071.jpg | Shopping | Travel | travel_booking | 0.3849 | Travel | **UNCHANGED** | Base Top-1 'Travel' retained (score=0.3034) |
| ext2_079 | ext2_079.jpg | Connectivity | General | travel_booking | 0.3223 | General | **UNCHANGED** | Base Top-1 'General' retained (score=0.3607) |
| ext2_084 | ext2_084.jpg | General | Education | code_editor_terminal | 0.3601 | Education | **UNCHANGED** | Base Top-1 'Education' retained (score=0.3926) |
| ext2_086 | ext2_086.jpg | Work & Career | Entertainment | code_editor_terminal | 0.2822 | Entertainment | **UNCHANGED** | Base Top-1 'Entertainment' retained (score=0.3644) |
| ext2_087 | ext2_087.jpg | Work & Career | General | code_editor_terminal | 0.2648 | General | **UNCHANGED** | Base Top-1 'General' retained (score=0.2400) |
| ext2_092 | ext2_092.jpg | General | Payment | travel_booking | 0.2880 | Payment | **UNCHANGED** | Base Top-1 'Payment' retained (score=0.6045) |
| ext2_093 | ext2_093.jpg | Education | Payment | travel_booking | 0.3211 | Payment | **UNCHANGED** | Base Top-1 'Payment' retained (score=0.5889) |
| ext2_094 | ext2_094.jpg | General | Payment | travel_booking | 0.3298 | Payment | **UNCHANGED** | Base Top-1 'Payment' retained (score=0.4779) |
| ext2_096 | ext2_096.jpg | General | Payment | travel_booking | 0.3580 | Payment | **UNCHANGED** | Base Top-1 'Payment' retained (score=0.4011) |
| ext2_098 | ext2_098.jpg | General | Payment | travel_booking | 0.3209 | Payment | **UNCHANGED** | Base Top-1 'Payment' retained (score=0.4814) |
| ext2_105 | ext2_105.jpg | General | Payment | travel_booking | 0.3915 | Payment | **UNCHANGED** | Base Top-1 'Payment' retained (score=0.4490) |
| ext2_106 | ext2_106.jpg | General | Payment | travel_booking | 0.3113 | Payment | **UNCHANGED** | Base Top-1 'Payment' retained (score=0.6380) |
| ext2_107 | ext2_107.jpg | General | Payment | travel_booking | 0.3217 | Payment | **UNCHANGED** | Base Top-1 'Payment' retained (score=0.6413) |
| ext2_108 | ext2_108.jpg | General | Payment | travel_booking | 0.3309 | Payment | **UNCHANGED** | Base Top-1 'Payment' retained (score=0.6151) |
| ext2_110 | ext2_110.jpg | Payment | Communication | travel_booking | 0.2878 | Communication | **UNCHANGED** | Base Top-1 'Communication' retained (score=0.2835) |
| ext2_111 | ext2_111.jpg | Education | Work & Career | document_article_view | 0.2730 | Work & Career | **UNCHANGED** | Base Top-1 'Work & Career' retained (score=0.4487) |
| ext2_113 | ext2_113.jpg | Education | General | dashboard_settings | 0.2409 | General | **UNCHANGED** | Base Top-1 'General' retained (score=0.5534) |
| ext2_114 | ext2_114.jpg | General | Work & Career | travel_booking | 0.3303 | Work & Career | **UNCHANGED** | Base Top-1 'Work & Career' retained (score=0.4236) |
| ext2_122 | ext2_122.jpg | Payment | Communication | media_player_library | 0.2642 | Communication | **UNCHANGED** | Base Top-1 'Communication' retained (score=0.5850) |
| ext2_126 | ext2_126.jpg | Education | Work & Career | travel_booking | 0.3656 | Work & Career | **UNCHANGED** | Base Top-1 'Work & Career' retained (score=0.5221) |
| ext2_127 | ext2_127.jpg | Education | Work & Career | travel_booking | 0.3011 | Work & Career | **UNCHANGED** | Base Top-1 'Work & Career' retained (score=0.5173) |
| ext2_128 | ext2_128.jpg | Education | Work & Career | travel_booking | 0.3486 | Work & Career | **UNCHANGED** | Base Top-1 'Work & Career' retained (score=0.6620) |
| ext2_130 | ext2_130.jpg | Communication | Work & Career | travel_booking | 0.2682 | Work & Career | **UNCHANGED** | Base Top-1 'Work & Career' retained (score=0.5146) |
| ext2_133 | ext2_133.jpg | Communication | Work & Career | media_player_library | 0.3322 | Work & Career | **UNCHANGED** | Base Top-1 'Work & Career' retained (score=0.6234) |
| ext2_138 | ext2_138.jpg | Education | Work & Career | travel_booking | 0.3840 | Work & Career | **UNCHANGED** | Base Top-1 'Work & Career' retained (score=0.4565) |
| ext2_139 | ext2_139.jpg | General | Travel | media_player_library | 0.3197 | Travel | **UNCHANGED** | Promoted 'Travel' (score=0.3048) over base Top-1 'General' via Multi-Signal Evidence |
| ext2_143 | ext2_143.jpg | Education | General | document_article_view | 0.3105 | General | **UNCHANGED** | Base Top-1 'General' retained (score=0.3476) |
| ext2_147 | ext2_147.jpg | Entertainment | General | media_player_library | 0.3448 | General | **UNCHANGED** | Base Top-1 'General' retained (score=0.3628) |
| ext2_150 | ext2_150.jpg | Communication | Work & Career | media_player_library | 0.3096 | Work & Career | **UNCHANGED** | Base Top-1 'Work & Career' retained (score=0.4680) |
| ext2_153 | ext2_153.jpg | Payment | Travel | travel_booking | 0.3709 | Travel | **UNCHANGED** | Base Top-1 'Travel' retained (score=0.4388) |
| ext2_158 | ext2_158.jpg | Communication | Work & Career | media_player_library | 0.3102 | Work & Career | **UNCHANGED** | Base Top-1 'Work & Career' retained (score=0.6379) |
| ext2_160 | ext2_160.jpg | General | Entertainment | media_player_library | 0.3841 | Entertainment | **UNCHANGED** | Base Top-1 'Entertainment' retained (score=0.4237) |
| ext2_162 | ext2_162.jpg | General | Travel | dashboard_settings | 0.2427 | Travel | **UNCHANGED** | Base Top-1 'Travel' retained (score=0.4525) |
| ext2_163 | ext2_163.jpg | Work & Career | Payment | media_player_library | 0.2486 | Payment | **UNCHANGED** | Base Top-1 'Payment' retained (score=0.3703) |
| ext2_180 | ext2_180.jpg | General | Entertainment | media_player_library | 0.2858 | Entertainment | **UNCHANGED** | Base Top-1 'Entertainment' retained (score=0.5316) |
| ext2_193 | ext2_193.jpg | General | Entertainment | media_player_library | 0.3436 | Entertainment | **UNCHANGED** | Base Top-1 'Entertainment' retained (score=0.3029) |
| ext2_200 | ext2_200.jpg | General | Entertainment | media_player_library | 0.3158 | Entertainment | **UNCHANGED** | Base Top-1 'Entertainment' retained (score=0.5435) |
| ext2_201 | ext2_201.jpg | General | Entertainment | travel_booking | 0.2826 | Entertainment | **UNCHANGED** | Base Top-1 'Entertainment' retained (score=0.5533) |
| ext2_202 | ext2_202.jpg | Shopping | General | travel_booking | 0.2591 | General | **UNCHANGED** | Base Top-1 'General' retained (score=0.3817) |
| ext2_206 | ext2_206.jpg | Education | Shopping | document_article_view | 0.2612 | Shopping | **UNCHANGED** | Base Top-1 'Shopping' retained (score=0.3743) |
| ext2_207 | ext2_207.jpg | General | Work & Career | travel_booking | 0.3394 | Work & Career | **UNCHANGED** | Base Top-1 'Work & Career' retained (score=0.3902) |
| ext2_210 | ext2_210.jpg | General | Education | media_player_library | 0.3900 | Education | **UNCHANGED** | Base Top-1 'Education' retained (score=0.3905) |
| ext2_211 | ext2_211.jpg | Education | Shopping | travel_booking | 0.2921 | Shopping | **UNCHANGED** | Base Top-1 'Shopping' retained (score=0.3671) |
| ext2_212 | ext2_212.jpg | General | Communication | media_player_library | 0.3134 | Communication | **UNCHANGED** | Base Top-1 'Communication' retained (score=0.2711) |
| ext2_213 | ext2_213.jpg | General | Communication | travel_booking | 0.2770 | Communication | **UNCHANGED** | Base Top-1 'Communication' retained (score=0.3575) |
| ext2_217 | ext2_217.jpg | General | Travel | travel_booking | 0.2473 | Travel | **UNCHANGED** | Base Top-1 'Travel' retained (score=0.4159) |
| ext2_224 | ext2_224.jpg | General | Communication | media_player_library | 0.2590 | Communication | **UNCHANGED** | Base Top-1 'Communication' retained (score=0.2785) |
| ext2_227 | ext2_227.jpg | General | Entertainment | travel_booking | 0.3510 | Entertainment | **UNCHANGED** | Base Top-1 'Entertainment' retained (score=0.4139) |
| ext2_228 | ext2_228.jpg | Communication | Entertainment | media_player_library | 0.2932 | Entertainment | **UNCHANGED** | Base Top-1 'Entertainment' retained (score=0.3014) |
| ext2_231 | ext2_231.jpg | Shopping | Work & Career | travel_booking | 0.3778 | Work & Career | **UNCHANGED** | Base Top-1 'Work & Career' retained (score=0.5135) |
| ext2_232 | ext2_232.jpg | Education | Work & Career | media_player_library | 0.2938 | Work & Career | **UNCHANGED** | Base Top-1 'Work & Career' retained (score=0.3963) |
| ext2_233 | ext2_233.jpg | General | Work & Career | media_player_library | 0.2562 | Work & Career | **UNCHANGED** | Base Top-1 'Work & Career' retained (score=0.4530) |
| ext2_237 | ext2_237.jpg | Shopping | Communication | media_player_library | 0.2789 | Communication | **UNCHANGED** | Base Top-1 'Communication' retained (score=0.3092) |
| ext2_240 | ext2_240.jpg | General | Payment | travel_booking | 0.2776 | Payment | **UNCHANGED** | Base Top-1 'Payment' retained (score=0.4149) |