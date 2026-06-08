# Translation vibe-check

Greedy (num_beams=1) CPU decoding. Qualitative sanity pass.


## Health summary

| run | kind | outputs | degenerate | degenerate frac | NaN param tensors |
|---|---|---|---|---|---|
| 360m-sentence | en-zh-causal | 12 | 1 | 0.083 | None |
| 360m-sentvocab | en-zh-causal | 12 | 3 | 0.25 | None |
| 1p7b-sentence | en-zh-causal | 12 | 1 | 0.083 | None |
| 1p7b-sentvocab | en-zh-causal | 12 | 1 | 0.083 | None |
| nllb-600m | nllb-seq2seq | 8 | 8 | 1.0 | 509 |

## 360m-sentence  (en-zh-causal)


### in_domain


**en2zh**

| source | hypothesis | reference |
|---|---|---|
| The president met with foreign leaders to discuss climate policy. | The president met with foreign leaders to discuss climate policy. |  |
| Scientists discovered a new species of fish in the deep ocean. | Scientists have discovered a new species of fish in the deep ocean. |  |
| The company reported strong earnings in the third quarter. | The company reported strong earnings in the third quarter. |  |

**zh2en**

| source | hypothesis | reference |
|---|---|---|
| 今天天气很好，我们去公园散步吧。 | The ಕಾಲವಾಗಿಸುವಾಗಿಸುವಾಗಿಸ |  |
| 中国的经济在过去几十年里增长迅速。 | P |  |
| 这家餐厅的菜很好吃，服务也很周到。 | ಠಠಠಠಠಠಠಠಠಠಠಠಠಠಠಠಠಠಠಠಠ |  |

### out_of_domain


**en2zh**

| source | hypothesis | reference |
|---|---|---|
| Break a leg at your performance tonight! | The following is a list of the most common questions we get asked about the program. |  |
| Configure the firewall to block inbound traffic on port 8080. | The firewall is configured to allow inbound traffic on port 8080. |  |
| The quarterback threw a Hail Mary in the final seconds of the game. | The quarterback threw a Hail Mary in the final seconds of the game. |  |

**zh2en**

| source | hypothesis | reference |
|---|---|---|
| 请在终端中运行该命令以重启服务器。 | K |  |
| 塞翁失马，焉知非福。 | The 1990s saw the rise of the 1990s, a decade of change. The 1990s saw the rise of the 1990s, a decade of change. |  |
| 这个函数的时间复杂度是 O(n log n)。 | Theorem 1.1. Let A be a set of n elements. Then, the following statements are equivalent: |  |

## 360m-sentvocab  (en-zh-causal)


### in_domain


**en2zh**

| source | hypothesis | reference |
|---|---|---|
| The president met with foreign leaders to discuss climate policy. | The president met with foreign leaders to discuss climate policy. |  |
| Scientists discovered a new species of fish in the deep ocean. | Scientists have discovered a new species of fish in the deep ocean. |  |
| The company reported strong earnings in the third quarter. | The company reported strong earnings in the third quarter. |  |

**zh2en**

| source | hypothesis | reference |
|---|---|---|
| 今天天气很好，我们去公园散步吧。 | The 100000000000000000000000000000000000000000000000000000000000 |  |
| 中国的经济在过去几十年里增长迅速。 | 100000000000000000000000000000000000000000000000000000000000000 |  |
| 这家餐厅的菜很好吃，服务也很周到。 | The 100000000000000000000000000000000000000000000000000000000000 |  |

### out_of_domain


**en2zh**

| source | hypothesis | reference |
|---|---|---|
| Break a leg at your performance tonight! | The following is a list of the most common questions we get asked about the program. |  |
| Configure the firewall to block inbound traffic on port 8080. | The firewall is configured to allow inbound traffic on port 8080. |  |
| The quarterback threw a Hail Mary in the final seconds of the game. | The quarterback threw a Hail Mary in the final seconds of the game. |  |

**zh2en**

| source | hypothesis | reference |
|---|---|---|
| 请在终端中运行该命令以重启服务器。 | K |  |
| 塞翁失马，焉知非福。 | The 1990s saw the rise of the 1990s, a decade of change. The 1990s saw the rise of the 1990s, a decade of change. |  |
| 这个函数的时间复杂度是 O(n log n)。 | Theorem 1.1. Let A be a set of n elements. Then, the following statements are equivalent: |  |

## 1p7b-sentence  (en-zh-causal)


### in_domain


**en2zh**

| source | hypothesis | reference |
|---|---|---|
| The president met with foreign leaders to discuss climate policy. | <\|tgt\|> <\|tgt\|> <\|tgt\|> <\|tgt\|>  he  said. <\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|> |  |
| Scientists discovered a new species of fish in the deep ocean. | <\|tgt\|> <\|tgt\|> <\|tgt\|> |  |
| The company reported strong earnings in the third quarter. | <\|tgt\|> <\|tgt\|> <\|tgt\|> |  |

**zh2en**

| source | hypothesis | reference |
|---|---|---|
| 今天天气很好，我们去公园散步吧。 | <\|endoftext\|> |  |
| 中国的经济在过去几十年里增长迅速。 | <\|endoftext\|><\|endoftext\|><\|endoftext\|> |  |
| 这家餐厅的菜很好吃，服务也很周到。 | <\|endoftext\|> |  |

### out_of_domain


**en2zh**

| source | hypothesis | reference |
|---|---|---|
| Break a leg at your performance tonight! | <\|tgt\|> <\|tgt\|> 』 |  |
| Configure the firewall to block inbound traffic on port 8080. | Action：<\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|> |  |
| The quarterback threw a Hail Mary in the final seconds of the game. | ，<\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|> |  |

**zh2en**

| source | hypothesis | reference |
|---|---|---|
| 请在终端中运行该命令以重启服务器。 | <\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|> |  |
| 塞翁失马，焉知非福。 | <\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|> |  |
| 这个函数的时间复杂度是 O(n log n)。 | ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， |  |

## 1p7b-sentvocab  (en-zh-causal)


### in_domain


**en2zh**

| source | hypothesis | reference |
|---|---|---|
| The president met with foreign leaders to discuss climate policy. | <\|endoftext\|> |  |
| Scientists discovered a new species of fish in the deep ocean. | <\|endoftext\|> |  |
| The company reported strong earnings in the third quarter. | <\|endoftext\|> |  |

**zh2en**

| source | hypothesis | reference |
|---|---|---|
| 今天天气很好，我们去公园散步吧。 | ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， ， |  |
| 中国的经济在过去几十年里增长迅速。 | <\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|> |  |
| 这家餐厅的菜很好吃，服务也很周到。 | ， <\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|> |  |

### out_of_domain


**en2zh**

| source | hypothesis | reference |
|---|---|---|
| Break a leg at your performance tonight! | <\|endoftext\|> |  |
| Configure the firewall to block inbound traffic on port 8080. | <\|endoftext\|> |  |
| The quarterback threw a Hail Mary in the final seconds of the game. | <\|endoftext\|> |  |

**zh2en**

| source | hypothesis | reference |
|---|---|---|
| 请在终端中运行该命令以重启服务器。 | <\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|> |  |
| 塞翁失马，焉知非福。 | ，  the，  the，  the，  the，  the，  the，  the，  the，  the，  the，  the，  the，  the，  the，  the，  the，  the，  the，  the，  the，  the， |  |
| 这个函数的时间复杂度是 O(n log n)。 | <\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|><\|endoftext\|> |  |

## nllb-600m  (nllb-seq2seq)


### in_domain


**es2nasa**

| source | hypothesis | reference |
|---|---|---|
| Esta carta viene de parte de Pablo, llamado para ser un apóstol de Jesucristo, conforme a la voluntad de Dios, y de parte de nuestro hermano Sóstenes. | ∅ | Adxa' Pablo yaaseth, Dxus peeygãhçxa pa'yate Jesukristo jxkaahnisa. Txã'wẽ yũuwa'ja's Dxusku txhitxhçxa nvxiht. |
| Es enviada a la iglesia de Dios en Corinto, a aquellos que han sido justificados en Cristo Jesús, llamados para vivir en santidad, y a todos los que adoran al Señor Jesús en todas partes, el Señor de ellos y de nosotros. | ∅ | Korinto çxhabte yakhthẽ'jwe'sxtxi weçxana fxi'jaçthu, txã'wẽy yakhthẽ'j Sosteneswa yuwe weçxaakaja'k. I'kwe'sxa' Kristo Jesus yakh txutemée fxi'zeya' utxaapa'ga Dxus luuçxi'kwe. Majũwe'sxwa kwe'sxtxi jxpe'jsa Jesukristo yasena kxsussa ũstxna txãawe'sx yã'jçxa ja'daçxah txajx tasxte yuuçxáa fxi'zekahnku Dxusa' pa'ya. Txãa Jesukristo' kwe'sx jxukaysatx jxpe'jsa'. |
| Reciban gracia y paz de parte de Dios, nuestro Padre, y del Señor Jesucristo. | ∅ | I'kwe'sxtxi' Dxusa' pu'çxhina vxite' kwe'sxtxi jxpe'jsa Jesukristowa. Txãa pa'ga Dxus yakhçxáa mfxi'zewe. Jesukristo pu'çxpa'ga Dxusa' kĩh yuhwa peejxmée jxukak pees |

**nasa2es**

| source | hypothesis | reference |
|---|---|---|
| Adxa' Pablo yaaseth, Dxus peeygãhçxa pa'yate Jesukristo jxkaahnisa. Txã'wẽ yũuwa'ja's Dxusku txhitxhçxa nvxiht. | ∅ | Esta carta viene de parte de Pablo, llamado para ser un apóstol de Jesucristo, conforme a la voluntad de Dios, y de parte de nuestro hermano Sóstenes. |
| Korinto çxhabte yakhthẽ'jwe'sxtxi weçxana fxi'jaçthu, txã'wẽy yakhthẽ'j Sosteneswa yuwe weçxaakaja'k. I'kwe'sxa' Kristo Jesus yakh txutemée fxi'zeya' utxaapa'ga Dxus luuçxi'kwe. Majũwe'sxwa kwe'sxtxi jxpe'jsa Jesukristo yasena kxsussa ũstxna txãawe'sx yã'jçxa ja'daçxah txajx tasxte yuuçxáa fxi'zekahnku Dxusa' pa'ya. Txãa Jesukristo' kwe'sx jxukaysatx jxpe'jsa'. | ∅ | Es enviada a la iglesia de Dios en Corinto, a aquellos que han sido justificados en Cristo Jesús, llamados para vivir en santidad, y a todos los que adoran al Señor Jesús en todas partes, el Señor de ellos y de nosotros. |
| I'kwe'sxtxi' Dxusa' pu'çxhina vxite' kwe'sxtxi jxpe'jsa Jesukristowa. Txãa pa'ga Dxus yakhçxáa mfxi'zewe. Jesukristo pu'çxpa'ga Dxusa' kĩh yuhwa peejxmée jxukak pees | ∅ | Reciban gracia y paz de parte de Dios, nuestro Padre, y del Señor Jesucristo. |

### out_of_domain


**es2nasa**

| source | hypothesis | reference |
|---|---|---|
| El gobierno aprobó una nueva ley económica. | ∅ |  |
| Por favor reinicie el servidor desde la terminal. | ∅ |  |
