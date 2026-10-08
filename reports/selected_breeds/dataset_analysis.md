# Comprehensive Dataset Analysis & Phase 1 Audit

**Project**: AI-Powered Intelligent Breed Recognition System for Indian Cattle and Buffaloes  
**Version**: Selected-Breed High-Accuracy Version  
**Audit Date**: 2026-10-02  
**Dataset Path**: `dataset/expanded_82_breeds_v2`  
**Total Images Inspected**: 589 real photographs  
**Total Breeds Cataloged**: 82 breeds (59 Cattle, 23 Buffalo)  

---

## 1. Executive Summary

An exhaustive physical file-level audit was conducted across all real photographs in the copied project repository.
- Total Real Photographs: **589**
- Usable Images (decodable, $\ge 200\times 200$): **570** (96.77%)
- Corrupted Images: **0**
- Total Exact Cryptographic Duplicates (SHA-256): **0**
- Near-Duplicate Cross-References ($dHash \le 3$): **8** pairs detected

---

## 2. Complete 82-Breed Real Image Inventory

| Breed Name | Species | Real Images | Usable Images | Exact Duplicates | Near Duplicates | Image Dimensions (Median, Min-Max) | Source |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **Manda** | Buffalo | 31 | 31 | 0 | 0 | Med: 4032x3024 (Min: 650x550, Max: 6456x4304) | ICAR-NBAGR / Baseline Archive |
| **Surti** | Buffalo | 21 | 20 | 0 | 0 | Med: 1080x1280 (Min: 213x160, Max: 4608x4320) | ICAR-NBAGR / Baseline Archive |
| **Toda** | Buffalo | 17 | 16 | 0 | 0 | Med: 800x800 (Min: 181x200, Max: 5150x4000) | ICAR-NBAGR / Baseline Archive |
| **Nagpuri** | Buffalo | 11 | 11 | 0 | 0 | Med: 1902x1952 (Min: 1136x720, Max: 4160x3000) | ICAR-NBAGR / Baseline Archive |
| **Murrah** | Buffalo | 10 | 10 | 0 | 0 | Med: 3552x2536 (Min: 654x530, Max: 5712x4284) | ICAR-NBAGR / Baseline Archive |
| **Mehsana** | Buffalo | 9 | 9 | 0 | 0 | Med: 3456x2304 (Min: 650x550, Max: 4096x3000) | ICAR-NBAGR / Baseline Archive |
| **Bargur** | Buffalo | 6 | 5 | 0 | 4 | Med: 650x550 (Min: 160x240, Max: 1600x1200) | ICAR-NBAGR / Baseline Archive |
| **Banni** | Buffalo | 3 | 3 | 0 | 0 | Med: 650x550 (Min: 650x550, Max: 5600x3728) | ICAR-NBAGR / Baseline Archive |
| **Gomanchali** | Buffalo | 3 | 3 | 0 | 0 | Med: 881x661 (Min: 881x661, Max: 881x661) | ICAR-NBAGR / Baseline Archive |
| **Nili Ravi** | Buffalo | 3 | 3 | 0 | 0 | Med: 650x550 (Min: 650x539, Max: 737x550) | ICAR-NBAGR / Baseline Archive |
| **Bhadawari** | Buffalo | 2 | 2 | 0 | 0 | Med: 650x550 (Min: 650x550, Max: 650x550) | ICAR-NBAGR / Baseline Archive |
| **Chhattisgarhi** | Buffalo | 2 | 2 | 0 | 0 | Med: 650x550 (Min: 650x550, Max: 650x550) | ICAR-NBAGR / Baseline Archive |
| **Chilika** | Buffalo | 2 | 2 | 0 | 0 | Med: 650x550 (Min: 650x550, Max: 650x550) | ICAR-NBAGR / Baseline Archive |
| **Dharwadi** | Buffalo | 2 | 2 | 0 | 0 | Med: 650x550 (Min: 650x550, Max: 650x550) | ICAR-NBAGR / Baseline Archive |
| **Gojri** | Buffalo | 2 | 2 | 0 | 0 | Med: 650x550 (Min: 650x550, Max: 650x550) | ICAR-NBAGR / Baseline Archive |
| **Jaffarabadi** | Buffalo | 2 | 2 | 0 | 0 | Med: 2947x1751 (Min: 1799x1198, Max: 4096x2304) | ICAR-NBAGR / Baseline Archive |
| **Kalahandi** | Buffalo | 2 | 2 | 0 | 0 | Med: 650x550 (Min: 650x550, Max: 650x550) | ICAR-NBAGR / Baseline Archive |
| **Luit (Swamp)** | Buffalo | 2 | 2 | 0 | 0 | Med: 650x550 (Min: 650x550, Max: 650x550) | ICAR-NBAGR / Baseline Archive |
| **Manah** | Buffalo | 2 | 2 | 0 | 0 | Med: 650x550 (Min: 650x550, Max: 650x550) | ICAR-NBAGR / Baseline Archive |
| **Marathwadi** | Buffalo | 2 | 1 | 0 | 0 | Med: 411x227 (Min: 360x175, Max: 462x279) | ICAR-NBAGR / Baseline Archive |
| **Melghati** | Buffalo | 2 | 2 | 0 | 0 | Med: 920x712 (Min: 815x635, Max: 1026x789) | ICAR-NBAGR / Baseline Archive |
| **Pandharpuri** | Buffalo | 2 | 2 | 0 | 0 | Med: 650x550 (Min: 650x550, Max: 650x550) | ICAR-NBAGR / Baseline Archive |
| **Purnathadi** | Buffalo | 2 | 2 | 0 | 0 | Med: 650x550 (Min: 650x550, Max: 650x550) | ICAR-NBAGR / Baseline Archive |
| **Dangi** | Cattle | 43 | 39 | 0 | 0 | Med: 558x452 (Min: 162x169, Max: 6016x3456) | Wikimedia Commons, ICAR-NBAGR / Baseline Archive |
| **Gaolao** | Cattle | 33 | 33 | 0 | 0 | Med: 2000x2048 (Min: 299x423, Max: 5250x4512) | ICAR-NBAGR / Baseline Archive, Wikimedia Commons |
| **Gir** | Cattle | 31 | 30 | 0 | 0 | Med: 4032x3044 (Min: 236x157, Max: 6123x4284) | ICAR-NBAGR / Baseline Archive |
| **Ongole** | Cattle | 29 | 29 | 0 | 0 | Med: 1620x1200 (Min: 231x287, Max: 5712x4284) | ICAR-NBAGR / Baseline Archive |
| **Bachaur** | Cattle | 23 | 23 | 0 | 0 | Med: 1128x776 (Min: 360x209, Max: 4608x4160) | Wikimedia Commons, ICAR-NBAGR / Baseline Archive |
| **Sahiwal** | Cattle | 22 | 22 | 0 | 0 | Med: 1850x1070 (Min: 305x254, Max: 5184x4000) | ICAR-NBAGR / Baseline Archive |
| **Siri** | Cattle | 21 | 21 | 0 | 0 | Med: 3072x2048 (Min: 640x428, Max: 6048x4624) | ICAR-NBAGR / Baseline Archive |
| **Vechur** | Cattle | 19 | 19 | 0 | 0 | Med: 4608x3456 (Min: 2048x1536, Max: 5184x3863) | ICAR-NBAGR / Baseline Archive |
| **Badri** | Cattle | 17 | 15 | 0 | 0 | Med: 742x840 (Min: 142x263, Max: 3712x3648) | ICAR-NBAGR / Baseline Archive |
| **Kangayam** | Cattle | 17 | 17 | 0 | 0 | Med: 2250x2368 (Min: 480x744, Max: 5184x6000) | ICAR-NBAGR / Baseline Archive |
| **Nagori** | Cattle | 17 | 17 | 0 | 0 | Med: 4032x3024 (Min: 488x300, Max: 5184x3648) | ICAR-NBAGR / Baseline Archive |
| **Punganur** | Cattle | 16 | 16 | 0 | 0 | Med: 2379x2582 (Min: 779x492, Max: 4608x4032) | ICAR-NBAGR / Baseline Archive |
| **Kankrej** | Cattle | 15 | 15 | 0 | 0 | Med: 3264x1836 (Min: 1080x790, Max: 5184x3456) | ICAR-NBAGR / Baseline Archive |
| **Nimari** | Cattle | 12 | 12 | 0 | 0 | Med: 1900x2930 (Min: 1150x1724, Max: 8192x5464) | ICAR-NBAGR / Baseline Archive |
| **Tharparkar** | Cattle | 11 | 11 | 0 | 0 | Med: 2967x2961 (Min: 1120x800, Max: 5184x4160) | ICAR-NBAGR / Baseline Archive |
| **Gangatiri** | Cattle | 8 | 8 | 0 | 0 | Med: 3647x2045 (Min: 2200x1233, Max: 5184x3456) | ICAR-NBAGR / Baseline Archive |
| **Bargur** | Cattle | 7 | 6 | 0 | 4 | Med: 1600x1200 (Min: 160x240, Max: 5184x3456) | ICAR-NBAGR / Baseline Archive, Wikimedia Commons |
| **Amritmahal** | Cattle | 6 | 6 | 0 | 0 | Med: 3803x2841 (Min: 1546x1036, Max: 5184x3456) | ICAR-NBAGR / Baseline Archive |
| **Hallikar** | Cattle | 6 | 6 | 0 | 0 | Med: 2245x2282 (Min: 513x598, Max: 5184x3456) | ICAR-NBAGR / Baseline Archive |
| **Periyar** | Cattle | 6 | 6 | 0 | 0 | Med: 750x788 (Min: 720x405, Max: 4320x4608) | ICAR-NBAGR / Baseline Archive |
| **Ladakhi** | Cattle | 5 | 5 | 0 | 0 | Med: 4240x2400 (Min: 4240x2400, Max: 4240x2400) | ICAR-NBAGR / Baseline Archive |
| **Red Sindhi** | Cattle | 5 | 5 | 0 | 0 | Med: 960x720 (Min: 719x475, Max: 5184x3456) | ICAR-NBAGR / Baseline Archive |
| **Deoni** | Cattle | 4 | 4 | 0 | 0 | Med: 3164x2389 (Min: 800x411, Max: 5184x3456) | ICAR-NBAGR / Baseline Archive, Wikimedia Commons |
| **Hariana** | Cattle | 3 | 3 | 0 | 0 | Med: 5184x3456 (Min: 3024x1931, Max: 5184x3456) | ICAR-NBAGR / Baseline Archive |
| **Koppal** | Cattle | 3 | 3 | 0 | 0 | Med: 1600x1200 (Min: 1600x1200, Max: 5184x3456) | ICAR-NBAGR / Baseline Archive |
| **Malvi** | Cattle | 3 | 3 | 0 | 0 | Med: 5184x3456 (Min: 3024x3456, Max: 5184x4032) | ICAR-NBAGR / Baseline Archive |
| **Medini** | Cattle | 3 | 3 | 0 | 0 | Med: 3000x2193 (Min: 2369x1814, Max: 3000x2469) | ICAR-NBAGR / Baseline Archive |
| **Nari** | Cattle | 3 | 3 | 0 | 0 | Med: 1440x1312 (Min: 1061x720, Max: 2496x1708) | ICAR-NBAGR / Baseline Archive |
| **Ponwar** | Cattle | 3 | 3 | 0 | 0 | Med: 5184x3456 (Min: 3456x2304, Max: 5184x3456) | ICAR-NBAGR / Baseline Archive |
| **Red Kandhari** | Cattle | 3 | 3 | 0 | 0 | Med: 4608x3456 (Min: 2958x2395, Max: 5184x3456) | ICAR-NBAGR / Baseline Archive |
| **Belahi** | Cattle | 2 | 1 | 0 | 0 | Med: 1161x860 (Min: 274x184, Max: 2048x1536) | ICAR-NBAGR / Baseline Archive |
| **Binjharpuri** | Cattle | 2 | 2 | 0 | 0 | Med: 1663x952 (Min: 360x237, Max: 2966x1668) | ICAR-NBAGR / Baseline Archive |
| **Dagri** | Cattle | 2 | 2 | 0 | 0 | Med: 650x459 (Min: 612x459, Max: 689x459) | ICAR-NBAGR / Baseline Archive |
| **Himachali Pahari** | Cattle | 2 | 2 | 0 | 0 | Med: 682x456 (Min: 682x455, Max: 682x457) | ICAR-NBAGR / Baseline Archive |
| **Kathani** | Cattle | 2 | 1 | 0 | 0 | Med: 576x343 (Min: 253x190, Max: 900x496) | ICAR-NBAGR / Baseline Archive |
| **Kenkatha** | Cattle | 2 | 2 | 0 | 0 | Med: 5184x3456 (Min: 5184x3456, Max: 5184x3456) | ICAR-NBAGR / Baseline Archive |
| **Kherigarh** | Cattle | 2 | 2 | 0 | 0 | Med: 5184x3456 (Min: 5184x3456, Max: 5184x3456) | ICAR-NBAGR / Baseline Archive |
| **Khillar** | Cattle | 2 | 1 | 0 | 0 | Med: 2730x1819 (Min: 277x182, Max: 5184x3456) | ICAR-NBAGR / Baseline Archive |
| **Konkan Kapila** | Cattle | 2 | 2 | 0 | 0 | Med: 595x446 (Min: 593x445, Max: 598x448) | ICAR-NBAGR / Baseline Archive |
| **Krishna Valley** | Cattle | 2 | 2 | 0 | 0 | Med: 5184x3456 (Min: 5184x3456, Max: 5184x3456) | ICAR-NBAGR / Baseline Archive |
| **Lakhimi** | Cattle | 2 | 2 | 0 | 0 | Med: 574x430 (Min: 574x430, Max: 574x430) | ICAR-NBAGR / Baseline Archive |
| **Mahakaushali** | Cattle | 2 | 2 | 0 | 0 | Med: 546x559 (Min: 492x327, Max: 600x791) | ICAR-NBAGR / Baseline Archive |
| **Malnad Gidda** | Cattle | 2 | 2 | 0 | 0 | Med: 5184x3456 (Min: 5184x3456, Max: 5184x3456) | ICAR-NBAGR / Baseline Archive |
| **Masilum** | Cattle | 2 | 1 | 0 | 0 | Med: 284x215 (Min: 268x190, Max: 300x240) | ICAR-NBAGR / Baseline Archive |
| **Mewati** | Cattle | 2 | 2 | 0 | 0 | Med: 330x269 (Min: 300x239, Max: 360x300) | ICAR-NBAGR / Baseline Archive |
| **Motu** | Cattle | 2 | 2 | 0 | 0 | Med: 2352x2824 (Min: 1440x2448, Max: 3264x3200) | ICAR-NBAGR / Baseline Archive |
| **Poda Thurpu** | Cattle | 2 | 2 | 0 | 0 | Med: 657x464 (Min: 619x464, Max: 696x464) | ICAR-NBAGR / Baseline Archive |
| **Pulikulam** | Cattle | 2 | 2 | 0 | 0 | Med: 4607x2871 (Min: 4080x2296, Max: 5134x3447) | ICAR-NBAGR / Baseline Archive |
| **Purnea** | Cattle | 2 | 2 | 0 | 0 | Med: 608x321 (Min: 535x315, Max: 682x327) | ICAR-NBAGR / Baseline Archive |
| **Rathi** | Cattle | 2 | 2 | 0 | 0 | Med: 5184x3456 (Min: 5184x3456, Max: 5184x3456) | ICAR-NBAGR / Baseline Archive |
| **Rohilkhandi** | Cattle | 2 | 2 | 0 | 0 | Med: 760x409 (Min: 736x345, Max: 785x474) | ICAR-NBAGR / Baseline Archive |
| **Sanchori** | Cattle | 2 | 1 | 0 | 0 | Med: 1376x1055 (Min: 193x190, Max: 2560x1920) | ICAR-NBAGR / Baseline Archive |
| **Shweta Kapila** | Cattle | 2 | 2 | 0 | 0 | Med: 830x398 (Min: 660x367, Max: 1000x429) | ICAR-NBAGR / Baseline Archive |
| **Thutho** | Cattle | 2 | 2 | 0 | 0 | Med: 569x401 (Min: 540x398, Max: 598x404) | ICAR-NBAGR / Baseline Archive |
| **Umarda** | Cattle | 2 | 2 | 0 | 0 | Med: 1200x680 (Min: 1200x675, Max: 1200x685) | ICAR-NBAGR / Baseline Archive |
| **Umblachery** | Cattle | 2 | 2 | 0 | 0 | Med: 4122x3768 (Min: 3060x3456, Max: 5184x4080) | ICAR-NBAGR / Baseline Archive |
| **Ghumusari** | Cattle | 1 | 0 | 0 | 0 | Med: 232x149 (Min: 232x149, Max: 232x149) | ICAR-NBAGR / Baseline Archive |
| **Khariar** | Cattle | 1 | 1 | 0 | 0 | Med: 360x270 (Min: 360x270, Max: 360x270) | ICAR-NBAGR / Baseline Archive |
| **Kosali** | Cattle | 1 | 0 | 0 | 0 | Med: 360x185 (Min: 360x185, Max: 360x185) | ICAR-NBAGR / Baseline Archive |

---

## 3. Key Observations for Breed Selection
1. **Severe Imbalance in 82 Classes**: 
   - Top breeds (e.g. Dangi, Gaolao, Gir, Manda, Ongole) have 29–43 real images.
   - Long-tail breeds (e.g. Kosali, Ghumusari, Khariar, Umblachery, Thutho) possess only 1–2 real images across the entire dataset.
   - Training an 82-class classifier on classes with 1–2 real images forced excessive reliance on synthetic artifacts and caused high confusion.
2. **Targeting High-Accuracy 8–10 Breed Cohort**:
   - Selecting 8 to 10 prominent breeds with $\ge 15-20+$ clean, high-resolution real photographs ensures viable train/validation/test partitions.
   - Balanced representation: e.g. 5–6 prominent Cattle breeds and 3–4 prominent Buffalo breeds.
   - Guarantees an independent, unseen locked test partition composed strictly of 100% real photographs with zero cross-split leakage.
