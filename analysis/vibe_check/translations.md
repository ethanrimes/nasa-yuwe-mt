# Translation vibe-check

Greedy (num_beams=1) CPU decoding. Qualitative sanity pass.


## Health summary

| run | kind | outputs | degenerate | degenerate frac | NaN param tensors |
|---|---|---|---|---|---|
| 360m-sentence | en-zh-causal | 12 | 0 | 0.0 | None |
| 360m-sentvocab | en-zh-causal | 12 | 0 | 0.0 | None |
| 1p7b-sentence | en-zh-causal | 12 | 3 | 0.25 | None |
| 1p7b-sentvocab | en-zh-causal | 12 | 4 | 0.333 | None |
| nllb-600m-sent | nllb-seq2seq | 8 | 0 | 0.0 | 0 |
| nllb-600m-sentvocab | nllb-seq2seq | 8 | 0 | 0.0 | 0 |
| nllb-1.3b-sent | nllb-seq2seq | 8 | 0 | 0.0 | 0 |
| nllb-1.3b-sentvocab | nllb-seq2seq | 8 | 0 | 0.0 | 0 |
| nllb-3.3b-sent | nllb-seq2seq | 8 | 0 | 0.0 | 0 |
| nllb-3.3b-sentvocab | nllb-seq2seq | 8 | 0 | 0.0 | 0 |

## 360m-sentence  (en-zh-causal)


### religious_bible


**en2zh**

| source | hypothesis | reference |
|---|---|---|
| Paul, called to be an apostle of Jesus Christ through the will of God, and our brother Sosthenes, | The first thing that we need to do is to understand the meaning of the word “apostle.” The word “apostle” is a Greek word that means “one sent out.” It is used in the New Testament to refer to the twelve disciples of Jesus, and to the twelve apostles of the | 奉神旨意，蒙召作耶稣基督使徒的保罗，同兄弟所提尼， |
| Grace to you and peace from God our Father and the Lord Jesus Christ. | The Lord is with you always. Amen. | 愿恩惠、平安从神我们的父并主耶稣基督归与你们。 |
| I always thank my God concerning you, for the grace of God which was given you in Christ Jesus; | The Greek word for grace is the same word used in the New Testament for the gift of the Holy Spirit. | 我常为你们感谢我的神，因神在基督耶稣里所赐给你们的恩惠； |

**zh2en**

| source | hypothesis | reference |
|---|---|---|
| 奉神旨意，蒙召作耶稣基督使徒的保罗，同兄弟所提尼， | The following is a translation of the first verse of the Song of Solomon, which is a love poem. | Paul, called to be an apostle of Jesus Christ through the will of God, and our brother Sosthenes, |
| 愿恩惠、平安从神我们的父并主耶稣基督归与你们。 | The first line of the poem is the opening line of the poem. The second line is the first line of the second stanza. The third line is the first line of the third stanza. The fourth line is the first line of the fourth stanza. The fifth line is the first line of the fifth stanza. | Grace to you and peace from God our Father and the Lord Jesus Christ. |
| 我常为你们感谢我的神，因神在基督耶稣里所赐给你们的恩惠； | The following is a list of the most common words in the English language. | I always thank my God concerning you, for the grace of God which was given you in Christ Jesus; |

### literary_cultural


**en2zh**

| source | hypothesis | reference |
|---|---|---|
| The umbilical cord of the newborn should be buried in the center of the hearth so that the boy or girl never suffers from stomach pain. | The umbilical cord should be buried in the center of the hearth so that the boy or girl never suffers from stomach pain. | 新生儿的脐带应该埋在炉灶的中心，这样男孩或女孩就永远不会遭受腹痛。 |
| The first haircut of the boy or girl must be kept with joy and capio corn, so that the hair is not damaged and to prevent baldness. | The hair must be cut in the middle of the head, and the hair must be cut in a way that it is not cut in the middle of the head. 2. The hair must be cut in the middle of the head, and the hair must be cut in a way that it is not cut | 男孩或女孩的第一次剪发应该带着喜悦和卡皮奥玉米一起保存，这样头发就不会受损，并能防止秃顶。 |
| If you want young women's childbirth to be quick, you must give them the hen's very first egg. | The first thing that the Lord did was to send a messenger to the king of Assyria, saying, "Go and get the king of Assyria, and bring him to me." 20 And the king of Assyria came to the king of Judah, and he took him to the king of Judah, and | 如果要让年轻女孩的分娩速度加快，必须给她们吃母鸡下的第一个蛋。 |

**zh2en**

| source | hypothesis | reference |
|---|---|---|
| 新生儿的脐带应该埋在炉灶的中心，这样男孩或女孩就永远不会遭受腹痛。 | The following is a list of the most common questions that we get asked about the program. | The umbilical cord of the newborn should be buried in the center of the hearth so that the boy or girl never suffers from stomach pain. |
| 男孩或女孩的第一次剪发应该带着喜悦和卡皮奥玉米一起保存，这样头发就不会受损，并能防止秃顶。 | , ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， | The first haircut of the boy or girl must be kept with joy and capio corn, so that the hair is not damaged and to prevent baldness. |
| 如果要让年轻女孩的分娩速度加快，必须给她们吃母鸡下的第一个蛋。 | The following is a list of the most common questions that we get asked about the product. | If you want young women's childbirth to be quick, you must give them the hen's very first egg. |

## 360m-sentvocab  (en-zh-causal)


### religious_bible


**en2zh**

| source | hypothesis | reference |
|---|---|---|
| Paul, called to be an apostle of Jesus Christ through the will of God, and our brother Sosthenes, | The first thing that we need to do is to understand the meaning of the word “apostle.” The word “apostle” is a Greek word that means “one sent out.” It is used in the New Testament to refer to the twelve disciples of Jesus, and to the twelve apostles of the | 奉神旨意，蒙召作耶稣基督使徒的保罗，同兄弟所提尼， |
| Grace to you and peace from God our Father and the Lord Jesus Christ. | The following is a list of the most common questions we get asked about the Bible. | 愿恩惠、平安从神我们的父并主耶稣基督归与你们。 |
| I always thank my God concerning you, for the grace of God which was given you in Christ Jesus; | The Greek word for grace is the same word used in the New Testament for the gift of the Holy Spirit. | 我常为你们感谢我的神，因神在基督耶稣里所赐给你们的恩惠； |

**zh2en**

| source | hypothesis | reference |
|---|---|---|
| 奉神旨意，蒙召作耶稣基督使徒的保罗，同兄弟所提尼， | The 1990s saw the rise of the 1990s, a decade of the 1990s, a decade of the 1990s, a decade of the 1990s, a decade of the 1990s | Paul, called to be an apostle of Jesus Christ through the will of God, and our brother Sosthenes, |
| 愿恩惠、平安从神我们的父并主耶稣基督归与你们。 | The first line of the poem is a reference to the Buddha, who is the source of all wisdom. The second line is a reference to the Buddha's teaching, which is the source of all wisdom. The third line is a reference to the Buddha's teaching, which is the source of all wisdom. The | Grace to you and peace from God our Father and the Lord Jesus Christ. |
| 我常为你们感谢我的神，因神在基督耶稣里所赐给你们的恩惠； | The following is a list of the most common words in the English language. | I always thank my God concerning you, for the grace of God which was given you in Christ Jesus; |

### literary_cultural


**en2zh**

| source | hypothesis | reference |
|---|---|---|
| The umbilical cord of the newborn should be buried in the center of the hearth so that the boy or girl never suffers from stomach pain. | The umbilical cord should be buried in the center of the hearth so that the boy or girl never suffers from stomach pain. | 新生儿的脐带应该埋在炉灶的中心，这样男孩或女孩就永远不会遭受腹痛。 |
| The first haircut of the boy or girl must be kept with joy and capio corn, so that the hair is not damaged and to prevent baldness. | The hair must be cut in the middle of the head, and the hair must be cut in a way that the hair is not cut in the middle of the head. | 男孩或女孩的第一次剪发应该带着喜悦和卡皮奥玉米一起保存，这样头发就不会受损，并能防止秃顶。 |
| If you want young women's childbirth to be quick, you must give them the hen's very first egg. | The first thing that the Lord Jesus Christ did was to give us the first of His commandments. He said, "Thou shalt have no other gods before Me." He said, "Thou shalt not make for yourself an idol." He said, "Thou shalt not take the name of the Lord thy | 如果要让年轻女孩的分娩速度加快，必须给她们吃母鸡下的第一个蛋。 |

**zh2en**

| source | hypothesis | reference |
|---|---|---|
| 新生儿的脐带应该埋在炉灶的中心，这样男孩或女孩就永远不会遭受腹痛。 | The following is a list of the most common questions we get asked about the product. | The umbilical cord of the newborn should be buried in the center of the hearth so that the boy or girl never suffers from stomach pain. |
| 男孩或女孩的第一次剪发应该带着喜悦和卡皮奥玉米一起保存，这样头发就不会受损，并能防止秃顶。 | D 主要的，主要的，主要的，主要的，主要的，主要的，主要的，主要的，主要的，主要的，主要的，主要的， | The first haircut of the boy or girl must be kept with joy and capio corn, so that the hair is not damaged and to prevent baldness. |
| 如果要让年轻女孩的分娩速度加快，必须给她们吃母鸡下的第一个蛋。 | The following is a list of the most common questions we get asked about the program. | If you want young women's childbirth to be quick, you must give them the hen's very first egg. |

## 1p7b-sentence  (en-zh-causal)


### religious_bible


**en2zh**

| source | hypothesis | reference |
|---|---|---|
| Paul, called to be an apostle of Jesus Christ through the will of God, and our brother Sosthenes, | <\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|> | 奉神旨意，蒙召作耶稣基督使徒的保罗，同兄弟所提尼， |
| Grace to you and peace from God our Father and the Lord Jesus Christ. | <\|tgt\|> ，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，， | 愿恩惠、平安从神我们的父并主耶稣基督归与你们。 |
| I always thank my God concerning you, for the grace of God which was given you in Christ Jesus; | <\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|> | 我常为你们感谢我的神，因神在基督耶稣里所赐给你们的恩惠； |

**zh2en**

| source | hypothesis | reference |
|---|---|---|
| 奉神旨意，蒙召作耶稣基督使徒的保罗，同兄弟所提尼， | ， 」 ， 」 ， 」 ， 」 ， 」 ， 」 ， 」 ， 」 ， 」 ， 」 ， 」 ， 」 ， | Paul, called to be an apostle of Jesus Christ through the will of God, and our brother Sosthenes, |
| 愿恩惠、平安从神我们的父并主耶稣基督归与你们。 | ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， | Grace to you and peace from God our Father and the Lord Jesus Christ. |
| 我常为你们感谢我的神，因神在基督耶稣里所赐给你们的恩惠； | <\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|> | I always thank my God concerning you, for the grace of God which was given you in Christ Jesus; |

### literary_cultural


**en2zh**

| source | hypothesis | reference |
|---|---|---|
| The umbilical cord of the newborn should be buried in the center of the hearth so that the boy or girl never suffers from stomach pain. | <\|tgt\|> <\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|> | 新生儿的脐带应该埋在炉灶的中心，这样男孩或女孩就永远不会遭受腹痛。 |
| The first haircut of the boy or girl must be kept with joy and capio corn, so that the hair is not damaged and to prevent baldness. | <\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|> | 男孩或女孩的第一次剪发应该带着喜悦和卡皮奥玉米一起保存，这样头发就不会受损，并能防止秃顶。 |
| If you want young women's childbirth to be quick, you must give them the hen's very first egg. | <\|tgt\|> <\|tgt\|> ， <\|endoftext\|> | 如果要让年轻女孩的分娩速度加快，必须给她们吃母鸡下的第一个蛋。 |

**zh2en**

| source | hypothesis | reference |
|---|---|---|
| 新生儿的脐带应该埋在炉灶的中心，这样男孩或女孩就永远不会遭受腹痛。 | Pharisees, and the scribes, and the chief priests, and the scribes, were gathered together. 24 And they questioned him, saying, "What then do you think? If we say, 'Jesus of Nazareth,' he will be handed over to the chief priests and scribes, and to be condemned. | The umbilical cord of the newborn should be buried in the center of the hearth so that the boy or girl never suffers from stomach pain. |
| 男孩或女孩的第一次剪发应该带着喜悦和卡皮奥玉米一起保存，这样头发就不会受损，并能防止秃顶。 | <\|tgt\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|> | The first haircut of the boy or girl must be kept with joy and capio corn, so that the hair is not damaged and to prevent baldness. |
| 如果要让年轻女孩的分娩速度加快，必须给她们吃母鸡下的第一个蛋。 | the  the  the  the  the  the  the  the  the  the  the  the  the  the  the  the  the  the  the  the  the  the  the  the  the  the  the  the  the  the  the  the | If you want young women's childbirth to be quick, you must give them the hen's very first egg. |

## 1p7b-sentvocab  (en-zh-causal)


### religious_bible


**en2zh**

| source | hypothesis | reference |
|---|---|---|
| Paul, called to be an apostle of Jesus Christ through the will of God, and our brother Sosthenes, | ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， | 奉神旨意，蒙召作耶稣基督使徒的保罗，同兄弟所提尼， |
| Grace to you and peace from God our Father and the Lord Jesus Christ. | <\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|> | 愿恩惠、平安从神我们的父并主耶稣基督归与你们。 |
| I always thank my God concerning you, for the grace of God which was given you in Christ Jesus; | ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， | 我常为你们感谢我的神，因神在基督耶稣里所赐给你们的恩惠； |

**zh2en**

| source | hypothesis | reference |
|---|---|---|
| 奉神旨意，蒙召作耶稣基督使徒的保罗，同兄弟所提尼， | ， 2000， 2000， 2000， 2000， 2000， 2000， 2000， 2000， 2000， 2000， 20 | Paul, called to be an apostle of Jesus Christ through the will of God, and our brother Sosthenes, |
| 愿恩惠、平安从神我们的父并主耶稣基督归与你们。 | ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， | Grace to you and peace from God our Father and the Lord Jesus Christ. |
| 我常为你们感谢我的神，因神在基督耶稣里所赐给你们的恩惠； | <\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|> | I always thank my God concerning you, for the grace of God which was given you in Christ Jesus; |

### literary_cultural


**en2zh**

| source | hypothesis | reference |
|---|---|---|
| The umbilical cord of the newborn should be buried in the center of the hearth so that the boy or girl never suffers from stomach pain. | <\|endoftext\|> | 新生儿的脐带应该埋在炉灶的中心，这样男孩或女孩就永远不会遭受腹痛。 |
| The first haircut of the boy or girl must be kept with joy and capio corn, so that the hair is not damaged and to prevent baldness. | <\|endoftext\|> | 男孩或女孩的第一次剪发应该带着喜悦和卡皮奥玉米一起保存，这样头发就不会受损，并能防止秃顶。 |
| If you want young women's childbirth to be quick, you must give them the hen's very first egg. | <\|endoftext\|> | 如果要让年轻女孩的分娩速度加快，必须给她们吃母鸡下的第一个蛋。 |

**zh2en**

| source | hypothesis | reference |
|---|---|---|
| 新生儿的脐带应该埋在炉灶的中心，这样男孩或女孩就永远不会遭受腹痛。 | <\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|> | The umbilical cord of the newborn should be buried in the center of the hearth so that the boy or girl never suffers from stomach pain. |
| 男孩或女孩的第一次剪发应该带着喜悦和卡皮奥玉米一起保存，这样头发就不会受损，并能防止秃顶。 | <\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|> | The first haircut of the boy or girl must be kept with joy and capio corn, so that the hair is not damaged and to prevent baldness. |
| 如果要让年轻女孩的分娩速度加快，必须给她们吃母鸡下的第一个蛋。 | 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 。 | If you want young women's childbirth to be quick, you must give them the hen's very first egg. |

## nllb-600m-sent  (nllb-seq2seq)


### in_domain


**es2nasa**

| source | hypothesis | reference |
|---|---|---|
| Esta carta viene de parte de Pablo, llamado para ser un apóstol de Jesucristo, conforme a la voluntad de Dios, y de parte de nuestro hermano Sóstenes. | Dxus yakhthẽ'jwe'sx jxkaahni's jxkaahni's jxkaahni's jxkaahni's jxkaahni's jxkaahni's jxkaahni's jxkaahni's jxkaahni's jxkaahni's jxkaahni's jxka | Adxa' Pablo yaaseth, Dxus peeygãhçxa pa'yate Jesukristo jxkaahnisa. Txã'wẽ yũuwa'ja's Dxusku txhitxhçxa nvxiht. |
| Es enviada a la iglesia de Dios en Corinto, a aquellos que han sido justificados en Cristo Jesús, llamados para vivir en santidad, y a todos los que adoran al Señor Jesús en todas partes, el Señor de ellos y de nosotros. | Dxusa' txãa pa'gatey txãa pa'gatey Dxus yakhthẽ'jwe'sx Korintos çxhabtewe'sxtxi txã'wẽ yũuwa'ja's pta'sxsa's. Txã'wẽ yũuwa'ja's Dxus yakh puutx we'weçxa Kristo Jesus | Korinto çxhabte yakhthẽ'jwe'sxtxi weçxana fxi'jaçthu, txã'wẽy yakhthẽ'j Sosteneswa yuwe weçxaakaja'k. I'kwe'sxa' Kristo Jesus yakh txutemée fxi'zeya' utxaapa'ga Dxus luuçxi'kwe. Majũwe'sxwa kwe'sxtxi jxpe'jsa Jesukristo yasena kxsussa ũstxna txãawe'sx yã'jçxa ja'daçxah txajx tasxte yuuçxáa fxi'zekahnku Dxusa' pa'ya. Txãa Jesukristo' kwe'sx jxukaysatx jxpe'jsa'. |
| Reciban gracia y paz de parte de Dios, nuestro Padre, y del Señor Jesucristo. | Dxusa' kwe'sxtxi kwe'sxtxi jxpe'jsa Jesukristo's kwe'sxtxi jxpe'jsa yu' jxpa'yakxna. | I'kwe'sxtxi' Dxusa' pu'çxhina vxite' kwe'sxtxi jxpe'jsa Jesukristowa. Txãa pa'ga Dxus yakhçxáa mfxi'zewe. Jesukristo pu'çxpa'ga Dxusa' kĩh yuhwa peejxmée jxukak pees |

**nasa2es**

| source | hypothesis | reference |
|---|---|---|
| Adxa' Pablo yaaseth, Dxus peeygãhçxa pa'yate Jesukristo jxkaahnisa. Txã'wẽ yũuwa'ja's Dxusku txhitxhçxa nvxiht. | Pablo, por la gracia de Dios y por la gracia de Jesucristo, | Esta carta viene de parte de Pablo, llamado para ser un apóstol de Jesucristo, conforme a la voluntad de Dios, y de parte de nuestro hermano Sóstenes. |
| Korinto çxhabte yakhthẽ'jwe'sxtxi weçxana fxi'jaçthu, txã'wẽy yakhthẽ'j Sosteneswa yuwe weçxaakaja'k. I'kwe'sxa' Kristo Jesus yakh txutemée fxi'zeya' utxaapa'ga Dxus luuçxi'kwe. Majũwe'sxwa kwe'sxtxi jxpe'jsa Jesukristo yasena kxsussa ũstxna txãawe'sx yã'jçxa ja'daçxah txajx tasxte yuuçxáa fxi'zekahnku Dxusa' pa'ya. Txãa Jesukristo' kwe'sx jxukaysatx jxpe'jsa'. | Saludos a los hermanos de Corinto, y a nuestro hermano Sosthénes, que está en Cristo Jesús, y que está en Dios por medio de nuestro Señor Jesucristo. | Es enviada a la iglesia de Dios en Corinto, a aquellos que han sido justificados en Cristo Jesús, llamados para vivir en santidad, y a todos los que adoran al Señor Jesús en todas partes, el Señor de ellos y de nosotros. |
| I'kwe'sxtxi' Dxusa' pu'çxhina vxite' kwe'sxtxi jxpe'jsa Jesukristowa. Txãa pa'ga Dxus yakhçxáa mfxi'zewe. Jesukristo pu'çxpa'ga Dxusa' kĩh yuhwa peejxmée jxukak pees | Que el Señor Jesucristo, nuestro Dios, os conceda la paz y la paz. | Reciban gracia y paz de parte de Dios, nuestro Padre, y del Señor Jesucristo. |

### out_of_domain


**es2nasa**

| source | hypothesis | reference |
|---|---|---|
| El gobierno aprobó una nueva ley económica. | Txã'wẽy yu'sa' txã'wẽy yu'sa' txã'wẽy yu'sa' txã'wẽy yu'. |  |
| Por favor reinicie el servidor desde la terminal. | Por favor, reinicie el servidor desde la terminal. |  |

## nllb-600m-sentvocab  (nllb-seq2seq)


### in_domain


**es2nasa**

| source | hypothesis | reference |
|---|---|---|
| Esta carta viene de parte de Pablo, llamado para ser un apóstol de Jesucristo, conforme a la voluntad de Dios, y de parte de nuestro hermano Sóstenes. | Dxus yakh fxi'zesa Jesukristo jxkaahnisa's Pablowe'sxtxi txã'wẽy yu' txã'wẽy yu' txã'wẽy yu' txã'wẽy yu' txã'wẽy yu' txã'wẽy yu' txã'wẽy yu' txã'wẽy yu' | Adxa' Pablo yaaseth, Dxus peeygãhçxa pa'yate Jesukristo jxkaahnisa. Txã'wẽ yũuwa'ja's Dxusku txhitxhçxa nvxiht. |
| Es enviada a la iglesia de Dios en Corinto, a aquellos que han sido justificados en Cristo Jesús, llamados para vivir en santidad, y a todos los que adoran al Señor Jesús en todas partes, el Señor de ellos y de nosotros. | Dxus yakh puutx we'weçxa txãa pa'gatey txã'wẽ yũuwa'ja's txã'wẽ yũuwa'ja's txã'wẽ yũuwa'ja's txã'wẽ yũuwa'ja's txã'wẽ yũuwa'ja's txã'wẽ yũuwa'ja's txã | Korinto çxhabte yakhthẽ'jwe'sxtxi weçxana fxi'jaçthu, txã'wẽy yakhthẽ'j Sosteneswa yuwe weçxaakaja'k. I'kwe'sxa' Kristo Jesus yakh txutemée fxi'zeya' utxaapa'ga Dxus luuçxi'kwe. Majũwe'sxwa kwe'sxtxi jxpe'jsa Jesukristo yasena kxsussa ũstxna txãawe'sx yã'jçxa ja'daçxah txajx tasxte yuuçxáa fxi'zekahnku Dxusa' pa'ya. Txãa Jesukristo' kwe'sx jxukaysatx jxpe'jsa'. |
| Reciban gracia y paz de parte de Dios, nuestro Padre, y del Señor Jesucristo. | Dxusa' kwe'sx Tata' kwe'sxtxi jxpe'jsa Jesukristo's ũuste jxpa'yakx. | I'kwe'sxtxi' Dxusa' pu'çxhina vxite' kwe'sxtxi jxpe'jsa Jesukristowa. Txãa pa'ga Dxus yakhçxáa mfxi'zewe. Jesukristo pu'çxpa'ga Dxusa' kĩh yuhwa peejxmée jxukak pees |

**nasa2es**

| source | hypothesis | reference |
|---|---|---|
| Adxa' Pablo yaaseth, Dxus peeygãhçxa pa'yate Jesukristo jxkaahnisa. Txã'wẽ yũuwa'ja's Dxusku txhitxhçxa nvxiht. | Pablo, enviado por la gracia de Dios y por la voluntad de Dios, y por la voluntad de Jesucristo. | Esta carta viene de parte de Pablo, llamado para ser un apóstol de Jesucristo, conforme a la voluntad de Dios, y de parte de nuestro hermano Sóstenes. |
| Korinto çxhabte yakhthẽ'jwe'sxtxi weçxana fxi'jaçthu, txã'wẽy yakhthẽ'j Sosteneswa yuwe weçxaakaja'k. I'kwe'sxa' Kristo Jesus yakh txutemée fxi'zeya' utxaapa'ga Dxus luuçxi'kwe. Majũwe'sxwa kwe'sxtxi jxpe'jsa Jesukristo yasena kxsussa ũstxna txãawe'sx yã'jçxa ja'daçxah txajx tasxte yuuçxáa fxi'zekahnku Dxusa' pa'ya. Txãa Jesukristo' kwe'sx jxukaysatx jxpe'jsa'. | Saludos a los hermanos de Corinto, y a nuestro hermano Sosthenes, a todos los que están en Cristo Jesús, a todos los que han sido sanados por Dios, y por el Señor Jesucristo, nuestro Dios y Salvador. | Es enviada a la iglesia de Dios en Corinto, a aquellos que han sido justificados en Cristo Jesús, llamados para vivir en santidad, y a todos los que adoran al Señor Jesús en todas partes, el Señor de ellos y de nosotros. |
| I'kwe'sxtxi' Dxusa' pu'çxhina vxite' kwe'sxtxi jxpe'jsa Jesukristowa. Txãa pa'ga Dxus yakhçxáa mfxi'zewe. Jesukristo pu'çxpa'ga Dxusa' kĩh yuhwa peejxmée jxukak pees | Que la gracia y la gracia de Dios, el Señor Jesucristo, sean con ustedes. | Reciban gracia y paz de parte de Dios, nuestro Padre, y del Señor Jesucristo. |

### out_of_domain


**es2nasa**

| source | hypothesis | reference |
|---|---|---|
| El gobierno aprobó una nueva ley económica. | Txã'wẽy yu'sa's ji'phmeesa's ji'phmeesa. |  |
| Por favor reinicie el servidor desde la terminal. | Txajũ's vxite's vxite's vxite's vxite's vxite's vxite's vxite's vxite's vxite's vxite's vxite's vxite's vxite's vxite's vxite |  |

## nllb-1.3b-sent  (nllb-seq2seq)


### in_domain


**es2nasa**

| source | hypothesis | reference |
|---|---|---|
| Esta carta viene de parte de Pablo, llamado para ser un apóstol de Jesucristo, conforme a la voluntad de Dios, y de parte de nuestro hermano Sóstenes. | Dxus yuwe's selpiiya' ãjãmeete', Jesukristo's selpiiya' ãjãwa'ja's Pablo, vxite' kwe'sx yakhthẽ'j Sostenes yakh txã'wẽ yũu. | Adxa' Pablo yaaseth, Dxus peeygãhçxa pa'yate Jesukristo jxkaahnisa. Txã'wẽ yũuwa'ja's Dxusku txhitxhçxa nvxiht. |
| Es enviada a la iglesia de Dios en Corinto, a aquellos que han sido justificados en Cristo Jesús, llamados para vivir en santidad, y a todos los que adoran al Señor Jesús en todas partes, el Señor de ellos y de nosotros. | Dxusa' txãawe'sx Kristo Jesus yakh fxi'zesawe'sxtxi txãawe'sxtxi txãawe'sxtxi txãawe'sxtxi txãawe'sxtxi txãawe'sxtxi txãawe'sxtxi txãawe'sxtxi txãawe'sxtxi tx | Korinto çxhabte yakhthẽ'jwe'sxtxi weçxana fxi'jaçthu, txã'wẽy yakhthẽ'j Sosteneswa yuwe weçxaakaja'k. I'kwe'sxa' Kristo Jesus yakh txutemée fxi'zeya' utxaapa'ga Dxus luuçxi'kwe. Majũwe'sxwa kwe'sxtxi jxpe'jsa Jesukristo yasena kxsussa ũstxna txãawe'sx yã'jçxa ja'daçxah txajx tasxte yuuçxáa fxi'zekahnku Dxusa' pa'ya. Txãa Jesukristo' kwe'sx jxukaysatx jxpe'jsa'. |
| Reciban gracia y paz de parte de Dios, nuestro Padre, y del Señor Jesucristo. | Kwe'sxtxi jxpe'jsa Jesukristowe'sx Dxus kwe'sx Tata's jxpa'gawa'j ji'phmeesaçxá's. | I'kwe'sxtxi' Dxusa' pu'çxhina vxite' kwe'sxtxi jxpe'jsa Jesukristowa. Txãa pa'ga Dxus yakhçxáa mfxi'zewe. Jesukristo pu'çxpa'ga Dxusa' kĩh yuhwa peejxmée jxukak pees |

**nasa2es**

| source | hypothesis | reference |
|---|---|---|
| Adxa' Pablo yaaseth, Dxus peeygãhçxa pa'yate Jesukristo jxkaahnisa. Txã'wẽ yũuwa'ja's Dxusku txhitxhçxa nvxiht. | Pablo, apóstol de Jesucristo, llamado por la gracia de Dios, y llamado por la buena noticia, | Esta carta viene de parte de Pablo, llamado para ser un apóstol de Jesucristo, conforme a la voluntad de Dios, y de parte de nuestro hermano Sóstenes. |
| Korinto çxhabte yakhthẽ'jwe'sxtxi weçxana fxi'jaçthu, txã'wẽy yakhthẽ'j Sosteneswa yuwe weçxaakaja'k. I'kwe'sxa' Kristo Jesus yakh txutemée fxi'zeya' utxaapa'ga Dxus luuçxi'kwe. Majũwe'sxwa kwe'sxtxi jxpe'jsa Jesukristo yasena kxsussa ũstxna txãawe'sx yã'jçxa ja'daçxah txajx tasxte yuuçxáa fxi'zekahnku Dxusa' pa'ya. Txãa Jesukristo' kwe'sx jxukaysatx jxpe'jsa'. | A todos los creyentes y santos que están en Corinto, a los que aman a Dios y a su Señor Jesucristo. Que Dios, nuestro Dios y Señor, les conceda paz y paz, | Es enviada a la iglesia de Dios en Corinto, a aquellos que han sido justificados en Cristo Jesús, llamados para vivir en santidad, y a todos los que adoran al Señor Jesús en todas partes, el Señor de ellos y de nosotros. |
| I'kwe'sxtxi' Dxusa' pu'çxhina vxite' kwe'sxtxi jxpe'jsa Jesukristowa. Txãa pa'ga Dxus yakhçxáa mfxi'zewe. Jesukristo pu'çxpa'ga Dxusa' kĩh yuhwa peejxmée jxukak pees | Que el Dios de la paz, el Dios de la gloria y nuestro Señor Jesucristo, os conceda paz y paz. | Reciban gracia y paz de parte de Dios, nuestro Padre, y del Señor Jesucristo. |

### out_of_domain


**es2nasa**

| source | hypothesis | reference |
|---|---|---|
| El gobierno aprobó una nueva ley económica. | Txã'wẽme'sx ley ũsçxa's ji'phme'sx. |  |
| Por favor reinicie el servidor desde la terminal. | Sa'sxa's serwera's endesa's pta'sxna. |  |

## nllb-1.3b-sentvocab  (nllb-seq2seq)


### in_domain


**es2nasa**

| source | hypothesis | reference |
|---|---|---|
| Esta carta viene de parte de Pablo, llamado para ser un apóstol de Jesucristo, conforme a la voluntad de Dios, y de parte de nuestro hermano Sóstenes. | Dxus jxkaahni's Jesukristo's selpiiya' ãjãsa's Pablo, vxite' kwe'sx yakhthẽ'j Sostenes yakh txãawe'sx. | Adxa' Pablo yaaseth, Dxus peeygãhçxa pa'yate Jesukristo jxkaahnisa. Txã'wẽ yũuwa'ja's Dxusku txhitxhçxa nvxiht. |
| Es enviada a la iglesia de Dios en Corinto, a aquellos que han sido justificados en Cristo Jesús, llamados para vivir en santidad, y a todos los que adoran al Señor Jesús en todas partes, el Señor de ellos y de nosotros. | Dxus luuçx Kristo Jesus yakh fxi'zewa'ja's pta'sxna fxi'jni's Corinto çxhabtewe'sxtxi txãawe'sxtxi txãawe'sxtxi txãawe'sxtxi. Txã'wẽ yũuçxa Dxus yakh fxi'zesawe | Korinto çxhabte yakhthẽ'jwe'sxtxi weçxana fxi'jaçthu, txã'wẽy yakhthẽ'j Sosteneswa yuwe weçxaakaja'k. I'kwe'sxa' Kristo Jesus yakh txutemée fxi'zeya' utxaapa'ga Dxus luuçxi'kwe. Majũwe'sxwa kwe'sxtxi jxpe'jsa Jesukristo yasena kxsussa ũstxna txãawe'sx yã'jçxa ja'daçxah txajx tasxte yuuçxáa fxi'zekahnku Dxusa' pa'ya. Txãa Jesukristo' kwe'sx jxukaysatx jxpe'jsa'. |
| Reciban gracia y paz de parte de Dios, nuestro Padre, y del Señor Jesucristo. | Kwe'sxtxi jxpe'jsa Jesukristowa kwe'sx Tata Dxus yakh puutx we'weni's. | I'kwe'sxtxi' Dxusa' pu'çxhina vxite' kwe'sxtxi jxpe'jsa Jesukristowa. Txãa pa'ga Dxus yakhçxáa mfxi'zewe. Jesukristo pu'çxpa'ga Dxusa' kĩh yuhwa peejxmée jxukak pees |

**nasa2es**

| source | hypothesis | reference |
|---|---|---|
| Adxa' Pablo yaaseth, Dxus peeygãhçxa pa'yate Jesukristo jxkaahnisa. Txã'wẽ yũuwa'ja's Dxusku txhitxhçxa nvxiht. | Esta carta viene de parte de Pablo, un apóstol de Jesucristo, quien por la voluntad de Dios, por su voluntad, fue llamado para ser apóstol. | Esta carta viene de parte de Pablo, llamado para ser un apóstol de Jesucristo, conforme a la voluntad de Dios, y de parte de nuestro hermano Sóstenes. |
| Korinto çxhabte yakhthẽ'jwe'sxtxi weçxana fxi'jaçthu, txã'wẽy yakhthẽ'j Sosteneswa yuwe weçxaakaja'k. I'kwe'sxa' Kristo Jesus yakh txutemée fxi'zeya' utxaapa'ga Dxus luuçxi'kwe. Majũwe'sxwa kwe'sxtxi jxpe'jsa Jesukristo yasena kxsussa ũstxna txãawe'sx yã'jçxa ja'daçxah txajx tasxte yuuçxáa fxi'zekahnku Dxusa' pa'ya. Txãa Jesukristo' kwe'sx jxukaysatx jxpe'jsa'. | A la iglesia de Dios que está en Corinto, y a todos los creyentes que están en Cristo Jesús. A todos los que han sido llamados hijos de Dios por medio de nuestro Señor Jesús. Que el Dios de nuestra fe y de nuestra paz les conceda su gracia y paz. | Es enviada a la iglesia de Dios en Corinto, a aquellos que han sido justificados en Cristo Jesús, llamados para vivir en santidad, y a todos los que adoran al Señor Jesús en todas partes, el Señor de ellos y de nosotros. |
| I'kwe'sxtxi' Dxusa' pu'çxhina vxite' kwe'sxtxi jxpe'jsa Jesukristowa. Txãa pa'ga Dxus yakhçxáa mfxi'zewe. Jesukristo pu'çxpa'ga Dxusa' kĩh yuhwa peejxmée jxukak pees | Que el Dios de la gloria y nuestro Señor Jesucristo os conceda paz y paz. | Reciban gracia y paz de parte de Dios, nuestro Padre, y del Señor Jesucristo. |

### out_of_domain


**es2nasa**

| source | hypothesis | reference |
|---|---|---|
| El gobierno aprobó una nueva ley económica. | Txã'wẽyçxáa yu'sa's ley yu'sa's. |  |
| Por favor reinicie el servidor desde la terminal. | Sa'sxna serwera's ksxa'sxna u'yçxa |  |

## nllb-3.3b-sent  (nllb-seq2seq)


### in_domain


**es2nasa**

| source | hypothesis | reference |
|---|---|---|
| Esta carta viene de parte de Pablo, llamado para ser un apóstol de Jesucristo, conforme a la voluntad de Dios, y de parte de nuestro hermano Sóstenes. | Adxa' Kristo Jesukristo's jxkaahnisawe'sxtxi txãawe'sx yakh fxi'zesa's jxpa'yakx, Pablo's vxite' Kristo's jxkaahnisawe'sxtxi txãawe'sx yakh fxi'zesa's txãawe'sx yakh | Adxa' Pablo yaaseth, Dxus peeygãhçxa pa'yate Jesukristo jxkaahnisa. Txã'wẽ yũuwa'ja's Dxusku txhitxhçxa nvxiht. |
| Es enviada a la iglesia de Dios en Corinto, a aquellos que han sido justificados en Cristo Jesús, llamados para vivir en santidad, y a todos los que adoran al Señor Jesús en todas partes, el Señor de ellos y de nosotros. | Idxa' Corinto çxhabte fxi'zesawe'sxtxi jxpe'jsa Jesukristo yakh fxi'zesawe'sxtxi txãawe'sx yakh fxi'zesawe'sxtxi txãawe'sx yakh fxi'zesawe'sx yakh fxi'zesawe'sx | Korinto çxhabte yakhthẽ'jwe'sxtxi weçxana fxi'jaçthu, txã'wẽy yakhthẽ'j Sosteneswa yuwe weçxaakaja'k. I'kwe'sxa' Kristo Jesus yakh txutemée fxi'zeya' utxaapa'ga Dxus luuçxi'kwe. Majũwe'sxwa kwe'sxtxi jxpe'jsa Jesukristo yasena kxsussa ũstxna txãawe'sx yã'jçxa ja'daçxah txajx tasxte yuuçxáa fxi'zekahnku Dxusa' pa'ya. Txãa Jesukristo' kwe'sx jxukaysatx jxpe'jsa'. |
| Reciban gracia y paz de parte de Dios, nuestro Padre, y del Señor Jesucristo. | Dxusa' kwe'sxtxi jxpe'jsa Jesukristo's jxpa'yakxçxa, Dxus yakh puutx we'wena fxi'zewa'ja's. | I'kwe'sxtxi' Dxusa' pu'çxhina vxite' kwe'sxtxi jxpe'jsa Jesukristowa. Txãa pa'ga Dxus yakhçxáa mfxi'zewe. Jesukristo pu'çxpa'ga Dxusa' kĩh yuhwa peejxmée jxukak pees |

**nasa2es**

| source | hypothesis | reference |
|---|---|---|
| Adxa' Pablo yaaseth, Dxus peeygãhçxa pa'yate Jesukristo jxkaahnisa. Txã'wẽ yũuwa'ja's Dxusku txhitxhçxa nvxiht. | Pablo, apóstol de Jesucristo por la gracia de Dios y por el llamado de Dios. | Esta carta viene de parte de Pablo, llamado para ser un apóstol de Jesucristo, conforme a la voluntad de Dios, y de parte de nuestro hermano Sóstenes. |
| Korinto çxhabte yakhthẽ'jwe'sxtxi weçxana fxi'jaçthu, txã'wẽy yakhthẽ'j Sosteneswa yuwe weçxaakaja'k. I'kwe'sxa' Kristo Jesus yakh txutemée fxi'zeya' utxaapa'ga Dxus luuçxi'kwe. Majũwe'sxwa kwe'sxtxi jxpe'jsa Jesukristo yasena kxsussa ũstxna txãawe'sx yã'jçxa ja'daçxah txajx tasxte yuuçxáa fxi'zekahnku Dxusa' pa'ya. Txãa Jesukristo' kwe'sx jxukaysatx jxpe'jsa'. | Saludos a la iglesia de Dios en Corinto, y a todos los que han sido llamados a ser santos en Cristo Jesús, y que han sido salvados por la fe en Dios y por nuestro Señor Jesucristo. | Es enviada a la iglesia de Dios en Corinto, a aquellos que han sido justificados en Cristo Jesús, llamados para vivir en santidad, y a todos los que adoran al Señor Jesús en todas partes, el Señor de ellos y de nosotros. |
| I'kwe'sxtxi' Dxusa' pu'çxhina vxite' kwe'sxtxi jxpe'jsa Jesukristowa. Txãa pa'ga Dxus yakhçxáa mfxi'zewe. Jesukristo pu'çxpa'ga Dxusa' kĩh yuhwa peejxmée jxukak pees | Que Dios, nuestro Padre y Señor Jesucristo os concedan gracia. Que Dios os conceda paz. | Reciban gracia y paz de parte de Dios, nuestro Padre, y del Señor Jesucristo. |

### out_of_domain


**es2nasa**

| source | hypothesis | reference |
|---|---|---|
| El gobierno aprobó una nueva ley económica. | El gobierno aprobó una nueva ley económica. |  |
| Por favor reinicie el servidor desde la terminal. | Por favor, reinicie el servidor desde el terminal. |  |

## nllb-3.3b-sentvocab  (nllb-seq2seq)


### in_domain


**es2nasa**

| source | hypothesis | reference |
|---|---|---|
| Esta carta viene de parte de Pablo, llamado para ser un apóstol de Jesucristo, conforme a la voluntad de Dios, y de parte de nuestro hermano Sóstenes. | Adxa' Jesukristo's nwẽese'jsa Pablo's nwẽese'jsa Jesukristo's nwẽese'jsa Jesukristo's nwẽese'jsawe'sx yakh fxi'zewa'ja's. | Adxa' Pablo yaaseth, Dxus peeygãhçxa pa'yate Jesukristo jxkaahnisa. Txã'wẽ yũuwa'ja's Dxusku txhitxhçxa nvxiht. |
| Es enviada a la iglesia de Dios en Corinto, a aquellos que han sido justificados en Cristo Jesús, llamados para vivir en santidad, y a todos los que adoran al Señor Jesús en todas partes, el Señor de ellos y de nosotros. | I'kwe'sxa' Corinto çxhabte fxi'zesawe'sxtxi jxpe'jsawe'sxtxi txãawe'sx jxpe'jsawe'sx txãawe'sx jxpe'jsawe'sx txãawe'sx jxpe'jsawe'sx txã | Korinto çxhabte yakhthẽ'jwe'sxtxi weçxana fxi'jaçthu, txã'wẽy yakhthẽ'j Sosteneswa yuwe weçxaakaja'k. I'kwe'sxa' Kristo Jesus yakh txutemée fxi'zeya' utxaapa'ga Dxus luuçxi'kwe. Majũwe'sxwa kwe'sxtxi jxpe'jsa Jesukristo yasena kxsussa ũstxna txãawe'sx yã'jçxa ja'daçxah txajx tasxte yuuçxáa fxi'zekahnku Dxusa' pa'ya. Txãa Jesukristo' kwe'sx jxukaysatx jxpe'jsa'. |
| Reciban gracia y paz de parte de Dios, nuestro Padre, y del Señor Jesucristo. | Dxusa' kwe'sxtxi jxpe'jsa Jesukristo's jxpa'yakxçxa txã'wẽ yũuwa'ja's ji'phmeesaçxáa. | I'kwe'sxtxi' Dxusa' pu'çxhina vxite' kwe'sxtxi jxpe'jsa Jesukristowa. Txãa pa'ga Dxus yakhçxáa mfxi'zewe. Jesukristo pu'çxpa'ga Dxusa' kĩh yuhwa peejxmée jxukak pees |

**nasa2es**

| source | hypothesis | reference |
|---|---|---|
| Adxa' Pablo yaaseth, Dxus peeygãhçxa pa'yate Jesukristo jxkaahnisa. Txã'wẽ yũuwa'ja's Dxusku txhitxhçxa nvxiht. | Pablo, apóstol de Cristo Jesús por la gracia de Dios y por el llamado de Dios. | Esta carta viene de parte de Pablo, llamado para ser un apóstol de Jesucristo, conforme a la voluntad de Dios, y de parte de nuestro hermano Sóstenes. |
| Korinto çxhabte yakhthẽ'jwe'sxtxi weçxana fxi'jaçthu, txã'wẽy yakhthẽ'j Sosteneswa yuwe weçxaakaja'k. I'kwe'sxa' Kristo Jesus yakh txutemée fxi'zeya' utxaapa'ga Dxus luuçxi'kwe. Majũwe'sxwa kwe'sxtxi jxpe'jsa Jesukristo yasena kxsussa ũstxna txãawe'sx yã'jçxa ja'daçxah txajx tasxte yuuçxáa fxi'zekahnku Dxusa' pa'ya. Txãa Jesukristo' kwe'sx jxukaysatx jxpe'jsa'. | Esta carta viene de parte de mi hermano Sostenes, y de la iglesia de Dios en Corinto, la cual ha sido llamada a ser santa en Cristo Jesús, y que está en el reino de Dios por medio de Jesucristo nuestro Señor y Señor. | Es enviada a la iglesia de Dios en Corinto, a aquellos que han sido justificados en Cristo Jesús, llamados para vivir en santidad, y a todos los que adoran al Señor Jesús en todas partes, el Señor de ellos y de nosotros. |
| I'kwe'sxtxi' Dxusa' pu'çxhina vxite' kwe'sxtxi jxpe'jsa Jesukristowa. Txãa pa'ga Dxus yakhçxáa mfxi'zewe. Jesukristo pu'çxpa'ga Dxusa' kĩh yuhwa peejxmée jxukak pees | La gracia de Dios y de nuestro Señor Jesucristo sea con ustedes. | Reciban gracia y paz de parte de Dios, nuestro Padre, y del Señor Jesucristo. |

### out_of_domain


**es2nasa**

| source | hypothesis | reference |
|---|---|---|
| El gobierno aprobó una nueva ley económica. | U'yuk txã'wẽy yu'sa's ya'pçxa's ya'pçxa's ya'pçxa's ya'pçxa's ya'pçxa'. |  |
| Por favor reinicie el servidor desde la terminal. | Txã'wẽyçxáatx pta'sxya's pta'sxya's pta'sxya's pta'sxya's pta'sxya's pta'sxya's pta'sxya's pta'sxya's pta'sx |  |
