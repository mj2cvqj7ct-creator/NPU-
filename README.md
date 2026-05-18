# Snapdragon X NPU Audio Enhancer

ARM64 Snapdragon X 搭載 PC の NPU を使い、Spotify、Apple Music、YouTube Music などの再生音を OS レベルで後処理して音質を改善するためのプロトタイプです。

## 重要な前提

- Spotify、Apple Music、YouTube Music のアプリ本体や暗号化された音源、推薦アルゴリズムを直接改変することはできません。
- 実装対象は、各アプリから OS に出力された PCM 音声を取得し、低遅延で補正して再生デバイスへ戻す「システムワイド音声エンハンサー」です。
- Snapdragon X の NPU は、ニューラル音声補正、軽量超解像、楽曲特徴抽出、ユーザー嗜好推定などの推論処理に使います。
- DRM、各サービスの利用規約、プライバシーを尊重し、録音保存やストリームの再配布は行いません。

## 現在の実装

このリポジトリには、実機の WASAPI/APO 接続や QNN SDK 依存を入れる前段階として、外部依存なしで検証できる C++17 のコア処理を追加しています。

- `src/dsp/`: 48 kHz stereo float ブロックのラウドネス解析、サービス別メイクアップゲイン、動的 EQ、トランジェント補正、true peak 風リミッター
- `src/inference/`: Snapdragon X NPU 向け QNN / ONNX Runtime QNN / DirectML / CPU fallback のバックエンド選択境界
- `src/profile/`: Spotify、Apple Music、YouTube Music ごとの安全な音質補正プロファイル
- `src/audio_capture/`: WASAPI loopback 実装を差し込むための capture interface
- `tests/`: DSP パイプラインとバックエンド選択の自動テスト

### ビルドとテスト

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

デモは合成音を処理し、選択されたバックエンドと処理前後の RMS/peak を表示します。

```bash
./build/sxnae_demo spotify
./build/sxnae_demo "Apple Music"
./build/sxnae_demo youtube_music
```

NPU ランタイムが組み込まれるまでは CPU fallback が選ばれます。実機統合時の選択テストには次の環境変数を使います。

```bash
SXNAE_ENABLE_QNN_HTP=1 ./build/sxnae_demo spotify
SXNAE_ENABLE_ONNX_QNN=1 ./build/sxnae_demo youtube_music
SXNAE_ENABLE_DIRECTML=1 ./build/sxnae_demo "Apple Music"
```

## 目標

1. 主要音楽サービスの出力音声に対して、アプリを問わず一貫した音質補正を行う。
2. ARM64 Windows 上で CPU 消費を抑え、NPU 推論により低遅延なリアルタイム処理を実現する。
3. ユーザーの聴取環境、ヘッドホン、音量、楽曲傾向に合わせて補正量を自動最適化する。
4. 推薦アルゴリズムそのものではなく、ローカルの聴取履歴特徴から「好みの音作り」を推定する。

## システム構成

```text
Music App
  -> Windows Audio Session / WASAPI
  -> Audio Capture Layer
  -> DSP Preprocess
  -> NPU Inference
  -> DSP Postprocess
  -> Audio Render Device
```

### Audio Capture Layer

- WASAPI loopback でアプリの出力をキャプチャします。
- 将来的には Windows Audio Processing Object (APO) として統合し、より自然なシステムワイド適用を目指します。
- 入力は 48 kHz / 32-bit float stereo を標準内部フォーマットにします。

### DSP Preprocess

- ラウドネス推定
- クリッピング検出
- 簡易ノイズフロア推定
- チャンネルバランス補正
- 楽曲区間の短時間スペクトル特徴抽出

### NPU Inference

Snapdragon X では、以下の順で実装候補を検討します。

1. Qualcomm AI Engine Direct SDK / QNN backend
2. ONNX Runtime の QNN Execution Provider
3. Windows ML / DirectML fallback
4. CPU fallback

推論モデルは小型化し、10 ms から 20 ms 程度のフレーム単位で動作させます。

現行コードでは `InferenceEngine` が入力音声の RMS、crest factor、clip 状態、サービス別プロファイルから `NeuralControls` を生成します。QNN/ONNX Runtime QNN 実装を追加する場合も、DSP 側には同じ control surface を渡すことで置き換えられます。

### DSP Postprocess

- ダイナミック EQ
- 軽量マルチバンドコンプレッション
- トランジェント保護
- ステレオ幅の過補正抑制
- true peak limiter

## 音質改善パイプライン

### Phase 1: ルールベース補正

- EBU R128 準拠のラウドネス正規化
- ヘッドホン別 EQ プロファイル
- true peak limiter
- 曲間音量差の低減

### Phase 2: NPU 支援の補正

- 楽曲ジャンル、音色、密度、低域量のローカル推定
- ボーカル帯域の明瞭度補正
- 過度な圧縮音源に対するトランジェント復元風補正
- 小音量再生時の聴感補正

### Phase 3: 個人化

- ユーザーが選んだプリセット、スキップ傾向、音量傾向をローカルで特徴化
- サービス横断で「好みの音作り」を学習
- 推薦サービスの内部ランキングは変更せず、ローカル体験のみを最適化

## サービス別の扱い

| サービス | 直接改変 | 実現可能な改善 |
| --- | --- | --- |
| Spotify | 不可 | OS 出力音声のリアルタイム補正、ローカル嗜好プロファイル |
| Apple Music | 不可 | ロスレス出力への後処理、ヘッドホン別補正 |
| YouTube Music | 不可 | ブラウザまたはアプリ出力の後処理、音量差補正 |

## 実装ロードマップ

1. WASAPI loopback の最小プロトタイプを作る。
2. 48 kHz stereo のリングバッファと低遅延 DSP チェーンを実装する。
3. ルールベースのラウドネス補正、EQ、limiter を追加する。
4. ONNX Runtime QNN Execution Provider で ARM64 / Snapdragon X NPU 推論を試す。
5. NPU が使えない環境では DirectML または CPU fallback に切り替える。
6. ローカル個人化プロファイルを暗号化保存する。
7. APO 化または仮想オーディオデバイス化して常用できる形にする。

## 評価指標

- エンドツーエンド遅延: 40 ms 未満を目標
- クリック、ポップノイズ、ドロップアウトなし
- true peak が 0 dBFS を超えない
- CPU 使用率を低く保ち、NPU 使用時のバッテリー影響を測定
- AB テストで補正オン/オフの主観評価を収集

## 非目標

- 配信サービスの DRM 回避
- 音源ファイルの保存、再配布、解析結果の外部送信
- Spotify、Apple Music、YouTube Music の推薦ランキングやアプリ内部ロジックの改変
- すべての音源を無条件に派手に加工すること

## 次に作るもの

- `src/audio_capture/`: `WasapiLoopbackCapture` の Windows 実装
- `src/inference/`: ONNX Runtime QNN Execution Provider と Qualcomm QNN HTP backend の実装
- `src/profile/`: ローカル個人化プロファイルの暗号化保存
- `tests/`: WAV 入出力による golden test と latency regression
- APO 化または仮想オーディオデバイス化
