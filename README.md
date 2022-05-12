# mpT5
mT5+Python


mT5(多言語モデル)から、英語・日本語以外の語彙を取り除くコード

参考：https://gist.github.com/avidale/44cd35bfcdaf8bedf51d97c468cc8001

# dataset
## dataset_maskfill

t5-smallモデル
 - t5_maxmin_pred.txt:予測候補(N=20)のリスト(二重リスト部分で失敗)
 - t5_maxmin_result.csv:正解・mask・予測結果(ナシ)

mt5-small
 - mT5_maxmin_pred.csv:予測候補(N=20)のリスト(○)
 - mT5_maxmin_result.csv:正解・mask・予測結果

mpT5卒論モデル
 - mpT5b_maxmin_pred.csv:予測候補(N=20)のリスト(○)
 - mpT5b_maxmin_result.csv:正解・mask・予測結果
