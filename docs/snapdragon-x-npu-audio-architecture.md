# Snapdragon X NPU Audio Architecture

この文書は、ARM64 Snapdragon X 搭載 PC で Spotify、Apple Music、YouTube Music などの出力音声を改善するための実装仕様です。配信サービスのアプリ、DRM、推薦ランキングには介入せず、OS に出力された PCM 音声を低遅延で後処理します。

## 基本方針

- 入力は WASAPI loopback または将来の APO から得られる 48 kHz / 32-bit float stereo PCM とする。
- NPU は音声を直接再生成するのではなく、補正パラメータ推定に使う。
- 実際のゲイン、EQ、コンプレッション、limiter は決定論的な DSP で行う。
- 解析結果、ユーザー嗜好、ヘッドホンプロファイルは端末内に閉じる。
- 推論失敗時も音量跳ねやクリックが出ないよう、DSP は常に安全な fallback 値を持つ。

## Runtime graph

```text
WASAPI loopback / APO
  -> format normalizer
  -> analysis ring buffer
  -> loudness and spectral feature extractor
  -> NPU inference scheduler
       -> Content Analyzer
       -> Enhancement Controller
       -> Preference Adapter
  -> smoothed control parameters
  -> deterministic DSP chain
       -> gain staging
       -> dynamic EQ
       -> multiband compression
       -> transient guard
       -> stereo safety
       -> true peak limiter
  -> render endpoint
```

## DSP chain

### Gain staging

- すべての処理前に -3 dBFS 程度の headroom を確保する。
- サービス間、曲間、広告や動画間の急な loudness 差を短期履歴でならす。
- integrated LUFS だけでは曲頭に間に合わないため、momentary / short-term loudness を併用する。

### Dynamic EQ

- バンドは初期実装で low shelf、low-mid、presence、air の 4 系統に絞る。
- NPU は各バンドの目標ゲインを直接決めず、補正方向と信頼度を返す。
- DSP 側で最大補正量、Q、attack/release を制限する。

### Multiband compression

- 主目的は音圧を上げることではなく、聴取音量に合わせて埋もれやすい帯域を整えること。
- 既に強くマスタリングされた音源では ratio と makeup gain を下げる。
- true peak limiter の負荷が増える設定は自動的に抑制する。

### Transient guard

- crest factor と帯域別 onset strength からアタック感の不足を推定する。
- 復元風の補正は控えめに行い、シンバルや歯擦音の過強調を避ける。
- YouTube Music など動画由来の音声では会話、効果音、広告にも反応するため、音楽らしさの信頼度が低い場合は無効化する。

### True peak limiter

- oversampling limiter を最終段に置く。
- ceiling は -1.0 dBTP を初期値にする。
- bypass、プリセット変更、NPU fallback 時にも limiter の状態を維持する。

## NPU model split

### Content Analyzer

目的は、曲や区間の性質を軽量に推定することです。

入力:

- 64 から 96 bin の log-mel spectrogram
- momentary loudness
- spectral centroid / rolloff
- crest factor
- stereo correlation

出力:

- music confidence
- vocal presence
- bass density
- treble harshness
- dynamic compression estimate
- transient strength

### Enhancement Controller

目的は、Analyzer の出力を安全な DSP 目標値へ変換することです。

入力:

- Analyzer embedding
- ユーザー選択プリセット
- ヘッドホンプロファイル
- 現在音量
- 直近の limiter gain reduction

出力:

- loudness target offset
- EQ band intent
- compression aggressiveness
- stereo width intent
- transient guard amount
- confidence score

Controller の出力はそのまま適用せず、DSP 側で以下を必ず適用します。

- 最大ゲイン制限
- 時間方向 smoothing
- limiter feedback による抑制
- confidence が低い場合の neutral blend

### Preference Adapter

目的は、サービス横断の「好みの音作り」を端末内で学習することです。

保存してよい情報:

- プリセット変更履歴
- 補正オン/オフの選択
- 音量帯の傾向
- 匿名化された補正パラメータ統計

保存しない情報:

- 曲名
- アーティスト名
- アルバム名
- 再生 URL
- 音声波形
- サービスアカウント情報

## Snapdragon X NPU execution

優先順位:

1. ONNX Runtime QNN Execution Provider
2. Qualcomm QNN backend への直接統合
3. DirectML fallback
4. CPU fallback

初期モデル要件:

- INT8 量子化を基本にする。
- 100 ms から 500 ms 間隔で推論する。
- 1 回の推論が 5 ms を超えたら fallback または推論頻度低下を検討する。
- DSP スレッドを NPU 推論待ちでブロックしない。

NPU が使える場合でも、低遅延安全性のために以下を守ります。

- Audio callback 内で推論を実行しない。
- 推論結果は lock-free queue または double buffer で DSP 側へ渡す。
- stale な結果には timestamp を付け、古すぎる場合は破棄する。
- 推論失敗、driver error、thermal throttling を通常状態として扱える設計にする。

## Service-specific behavior

| サービス | 典型的な入力経路 | 初期補正方針 |
| --- | --- | --- |
| Spotify | Desktop app / browser | loudness の一貫性、低域過多の抑制、プリセット反映 |
| Apple Music | Desktop app / browser | ロスレス出力を崩さない控えめな EQ、true peak 保護 |
| YouTube Music | Browser / PWA | 動画由来の音量差補正、広告や会話区間の過補正抑制 |

サービス識別は WASAPI session metadata で取得できる場合のみ使います。取得できない場合は音声特徴量とユーザー設定だけで動作します。

## Quality gates

実装ごとに次の自動チェックを用意します。

- WAV fixture の peak / true peak が制限内に収まる。
- DSP bypass と処理後の長さが一致する。
- 無音、低音量、最大音量近傍の入力で NaN / inf が出ない。
- プリセット変更時に 10 ms 以上の不連続ジャンプがない。
- NPU fallback 切替時に loudness が急変しない。

手動評価では次を確認します。

- Spotify、Apple Music、YouTube Music で曲間音量差が減る。
- ボーカルが前に出る設定でも歯擦音が痛くならない。
- 低音補正で limiter が常時動作しない。
- Bluetooth ヘッドホンで stereo widening が不自然にならない。
- NPU 使用時に CPU 使用率とバッテリー消費が CPU-only より改善する。

## Initial implementation slices

1. Offline WAV processor
   - format normalizer
   - loudness meter
   - fixed EQ
   - true peak limiter

2. Real-time engine
   - ring buffer
   - stable DSP parameter smoothing
   - WASAPI loopback prototype

3. NPU controller prototype
   - dummy ONNX model
   - QNN EP initialization
   - DirectML / CPU fallback

4. Personalization
   - local profile schema
   - preset statistics
   - encrypted storage

5. Product integration
   - tray UI
   - endpoint selection
   - APO or virtual audio device packaging
