# Snapdragon X NPU Execution Design

このメモは ARM64 Windows / Snapdragon X 搭載 PC で、音楽再生のリアルタイム後処理を NPU に載せるための実行設計です。Spotify、Apple Music、YouTube Music のアプリや配信ストリームには介入せず、OS が出力した PCM 音声だけを対象にします。

## 実行方針

1. WASAPI loopback または APO で 48 kHz / 32-bit float stereo の PCM を受け取る。
2. CPU SIMD で軽量な前処理を行い、モデル入力用の特徴量を作る。
3. ONNX Runtime QNN Execution Provider を第一候補として Snapdragon X NPU に推論を配置する。
4. 推論結果は音声波形そのものではなく、DSP 制御パラメータとして使う。
5. 音声サンプル単位の安全性は CPU 側の DSP と limiter で保証する。

NPU が直接 PCM 波形を全量変換する構成は、遅延、ドロップアウト、音質破綻時のリスクが大きいため初期実装では避けます。NPU は「何をどれくらい補正するか」を高速に判断し、実際の連続音声処理は決定論的な DSP に任せます。

## 推論ランタイム優先順位

| 優先度 | Backend | 用途 | 備考 |
| --- | --- | --- | --- |
| 1 | ONNX Runtime QNN EP | Snapdragon X NPU 推論 | 配布しやすく、モデル管理を ONNX に集約できる |
| 2 | Qualcomm AI Engine Direct / QNN SDK | 性能検証、詳細最適化 | EP で不足する op や量子化制御が必要な場合 |
| 3 | Windows ML / DirectML | NPU 非対応端末の GPU fallback | レイテンシより互換性を優先 |
| 4 | CPU | 安全 fallback | 補正量を控えめにし、ドロップアウトを防ぐ |

## フレーム設計

- Audio callback block: 5 ms から 10 ms
- Analysis window: 20 ms から 40 ms
- Feature hop: 10 ms
- NPU inference cadence: 20 ms から 50 ms
- DSP parameter smoothing: 100 ms から 500 ms

NPU 推論は audio callback のリアルタイムスレッドから直接呼ばず、別スレッドで非同期実行します。audio callback は最後に成功した推論結果を lock-free snapshot として読むだけにし、推論遅延が発生しても音切れしない構成にします。

```text
Audio callback
  -> append PCM to ring buffer
  -> read latest enhancement profile
  -> run deterministic DSP
  -> true peak limiter

Inference worker
  -> read analysis frames from ring buffer
  -> compute mel/STFT features
  -> run ONNX Runtime QNN EP
  -> publish smoothed DSP parameters
```

## NPU モデルの役割

### 1. Content classifier

楽曲の局所的な性質を推定します。

- vocal presence
- bass density
- transient density
- brightness
- compression / loudness saturation
- speech-like content
- silence or low-confidence region

出力は 0.0 から 1.0 の連続値とし、ジャンル名などの強いラベルには依存しません。

### 2. Enhancement policy model

ユーザー設定、デバイスプロファイル、content classifier の結果から、DSP パラメータの目標値を出します。

- dynamic EQ band gains
- multiband compressor threshold / ratio offset
- stereo width ceiling
- transient restoration amount
- low-volume compensation amount
- limiter headroom target

### 3. Personal tone embedding

ユーザーが選ぶプリセット、音量傾向、補正オン/オフの選択から、サービス横断のローカル音作りを表す小さな埋め込みを作ります。Spotify、Apple Music、YouTube Music の内部推薦アルゴリズムやランキングには影響させません。

## モデル仕様

| 項目 | 初期値 |
| --- | --- |
| Input sample rate | 48 kHz |
| Channels | stereo から mid/side と mono summary を生成 |
| Feature | log-mel, spectral centroid, crest factor, loudness, true peak margin |
| Quantization | int8 weight / int8 activation を優先 |
| Max model size | 5 MB 未満 |
| Target inference time | 5 ms 未満 |
| Failure behavior | 直前の安定パラメータを保持し、徐々に neutral profile へ戻す |

## リアルタイム安全性

- 推論 worker は audio callback を block しない。
- 推論結果には最大補正量、最大変化率、最小 headroom を設定する。
- confidence が低い場合は補正を弱める。
- true peak limiter は NPU 結果に関係なく常時有効にする。
- NPU 初期化失敗時はルールベース DSP のみで動作する。

## Snapdragon X 向け最適化

- ONNX graph は Conv1D/Conv2D、GEMM、LayerNorm など QNN EP が扱いやすい op に寄せる。
- dynamic shape を避け、固定 window / fixed batch にする。
- 量子化は calibration 音源セットで行い、過度な高域補正が出ないか AB 評価する。
- 推論 cadence を音質より電力効率側に寄せ、同じ区間で同一特徴が続く場合は inference skip を許可する。
- ARM64 CPU 側は特徴抽出と DSP に集中させ、FFT や IIR filter は既存の最適化ライブラリを使う。

## 実装順序

1. CPU only の feature extractor と rule-based DSP を実装する。
2. ONNX の dummy policy model を接続し、QNN EP / CPU fallback の切替をテストする。
3. content classifier を小型モデルに置き換える。
4. personal tone embedding をローカル保存に接続する。
5. APO または仮想オーディオデバイスに統合する。
