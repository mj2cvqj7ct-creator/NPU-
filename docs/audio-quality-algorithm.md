# 音質改善アルゴリズム設計

この文書は、Spotify、Apple Music、YouTube Music などの OS 出力 PCM に対して、サービス非依存に適用する音質改善アルゴリズムを定義します。各サービスの音源、DRM、推薦ランキング、アプリ内部ロジックは変更しません。

## 基本方針

- 補正対象は 48 kHz / 32-bit float / stereo の内部バッファ。
- 10 ms または 20 ms フレームで処理し、隣接フレームをオーバーラップさせる。
- NPU は「音を直接派手に加工する装置」ではなく、DSP パラメータを安定して推定する制御器として使う。
- true peak limiter を最終段に置き、0 dBFS 超過、ポップノイズ、過度なステレオ拡張を防ぐ。
- 個人化はローカル端末内で完結し、音声や特徴量を外部送信しない。

## パイプライン

```text
PCM input
  -> frame slicing
  -> analysis features
  -> rule-based safety checks
  -> NPU scene and preference inference
  -> parameter smoothing
  -> corrective DSP
  -> limiter
  -> PCM output
```

## 解析特徴量

DSP 前処理で以下を計算し、NPU 入力とルールベース補正の両方に使います。

| 特徴量 | 用途 |
| --- | --- |
| LUFS short-term / momentary | 曲間音量差の抑制、ゲイン上限設定 |
| true peak estimate | limiter の先読み量と出力天井 |
| spectral centroid | 明るさ、こもり、過度な高域の推定 |
| spectral flux | トランジェント量、圧縮感の推定 |
| band energy | 低域、中域、ボーカル帯域、高域の補正 |
| stereo correlation | ステレオ幅補正の安全判定 |
| silence / low energy ratio | 無音や曲間での誤学習防止 |

## NPU 推定タスク

### 1. コンテンツ分類

軽量 CNN または Conformer 風の小型モデルで、短い時間窓から音楽シーンを分類します。

- vocal-forward
- bass-heavy
- acoustic
- dense-master
- low-bitrate-artifact
- speech / podcast-like
- silence / transition

分類結果は EQ、compressor、transient restoration の補正量を決めるために使います。

### 2. 音質劣化推定

圧縮アーティファクト、過度なラウドネス、低域の膨らみ、ボーカル帯域のマスキングを推定します。出力は連続値にし、補正量を急変させないようにします。

```text
artifact_score: 0.0 - 1.0
masking_score: 0.0 - 1.0
harshness_score: 0.0 - 1.0
bass_bloat_score: 0.0 - 1.0
transient_loss_score: 0.0 - 1.0
```

### 3. 個人化プリファレンス推定

ユーザー操作から、音源ではなく音作りの好みを学習します。

- よく選ぶプリセット
- 手動 EQ 変更
- 音量変更の傾向
- 補正 ON/OFF の AB 選択
- サービス別ではなく端末共通の傾向

この推定は推薦アルゴリズムではなく、ローカル DSP パラメータの初期値にだけ使います。

## 補正モジュール

### Loudness normalizer

- EBU R128 風の short-term loudness を使う。
- 曲中のゲイン変化は 0.5 dB/s 以下を基本にする。
- 低音量の楽曲を上げる場合も、true peak 余裕を優先する。

### Dynamic EQ

NPU 推定とルールベース特徴量から、最大補正幅を制限した dynamic EQ を適用します。

| 帯域 | 主な目的 | 最大補正目安 |
| --- | --- | --- |
| 40-120 Hz | 低域の膨らみ抑制、薄い音の補強 | +/- 3 dB |
| 180-350 Hz | こもり抑制 | -2 dB |
| 1.5-4 kHz | ボーカル明瞭度 | +/- 2 dB |
| 6-10 kHz | 刺さり抑制、空気感 | +/- 2 dB |

### Transient protection / restoration

- transient_loss_score が高い場合だけ軽い補正を行う。
- attack を強調しすぎると疲れるため、percussive 成分の短時間ゲインに限定する。
- limiter 直前でピーク再確認を行う。

### Stereo image guard

- stereo correlation が低い音源では幅拡張を禁止する。
- ボーカル中心成分は mid を優先し、side の過補正を避ける。
- モノラル互換性を評価指標に含める。

### True peak limiter

- 最終出力は -1.0 dBTP を標準上限にする。
- 先読みありの軽量 limiter を使い、ドロップアウト時は bypass ではなく safe gain に戻す。

## サービス別プリセット

| サービス | 想定入力 | 初期補正 |
| --- | --- | --- |
| Spotify | 非可逆圧縮、音量正規化済みの場合あり | artifact_score と loudness を控えめに反映 |
| Apple Music | ロスレス/高品質出力の可能性 | EQ とヘッドホン補正を中心にし、復元系は弱め |
| YouTube Music | ブラウザ音声、動画由来の音量差 | loudness normalizer と harshness 抑制を強め |

サービス識別はプロセス名やセッション名のローカル判定に限定し、アプリ内部 API やストリーム URL には触れません。

## パラメータ平滑化

NPU 出力をそのまま DSP に接続すると音色が揺れるため、以下を適用します。

- attack: 50-150 ms
- release: 500-1500 ms
- scene change hold: 1-3 s
- maximum EQ delta per second: 1 dB/s

## 安全条件

- 無音、曲間、通知音では学習と強い補正を停止する。
- 入力がクリップしている場合、まずゲインを下げる。
- NPU 推論が遅延したフレームでは直近の安定パラメータを再利用する。
- 推論結果の信頼度が低い場合は Phase 1 のルールベース補正だけに戻す。

## テスト観点

- 1 kHz sine、pink noise、sweep でゲインとピークを確認する。
- 低ビットレート相当、過度に loud な音源、静かなクラシック、低域過多の EDM を WAV fixture にする。
- 補正 ON/OFF で true peak、LUFS、stereo correlation、遅延を自動測定する。
- AB テストでは「劇的」な変化だけでなく、長時間聴取の疲れにくさを評価する。
