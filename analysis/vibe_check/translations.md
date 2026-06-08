# Translation vibe-check

Greedy (num_beams=1) CPU decoding. Qualitative sanity pass.


## Health summary

| run | kind | outputs | degenerate | degenerate frac | NaN param tensors |
|---|---|---|---|---|---|
| nllb-600m-sent | nllb-seq2seq | 8 | 0 | 0.0 | 0 |
| nllb-600m-sentvocab | nllb-seq2seq | 8 | 0 | 0.0 | 0 |

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
