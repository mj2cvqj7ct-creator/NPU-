# Snapdragon X NPU Audio Enhancer

ARM64 Snapdragon X 搭載 PC の NPU を使い、Spotify、Apple Music、YouTube Music などの再生音を OS レベルで後処理して音質を改善するための設計メモと最小プロトタイプです。

## 重要な前提

- Spotify、Apple Music、YouTube Music のアプリ本体や暗号化された音源、推薦アルゴリズムを直接改変することはできません。
- 実装対象は、各アプリから OS に出力された PCM 音声を取得し、低遅延で補正して再生デバイスへ戻す「システムワイド音声エンハンサー」です。
- Snapdragon X の NPU は、ニューラル音声補正、軽量超解像、楽曲特徴抽出、ユーザー嗜好推定などの推論処理に使います。
- DRM、各サービスの利用規約、プライバシーを尊重し、録音保存やストリームの再配布は行いません。

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

## 現在のプロトタイプ

このリポジトリには、設計の最初の実装として Python 標準ライブラリだけで動く WAV オフライン処理プロトタイプを含めています。

- `src/snapdragon_npu_audio_enhancer/dsp.py`: DC 除去、チャンネルバランス、ラウドネス寄せ、簡易ダイナミック EQ、ステレオ幅制御、true peak limiter
- `src/snapdragon_npu_audio_enhancer/service_profiles.py`: Spotify、Apple Music、YouTube Music 向けの安全な後処理プリセット
- `src/snapdragon_npu_audio_enhancer/inference.py`: ONNX Runtime QNN Execution Provider を検出できる `SnapdragonNpuProvider` と、非 Snapdragon 環境でも動く heuristic fallback
- `src/snapdragon_npu_audio_enhancer/cli.py`: mono/stereo PCM WAV を読み、サービス別プロファイルと推論プロバイダを使って stereo WAV を出力する CLI

実行例:

```bash
python3 -m snapdragon_npu_audio_enhancer.cli input.wav output.wav --service spotify
python3 -m snapdragon_npu_audio_enhancer.cli --input input.wav --output output.wav --service youtube_music --onnx-model model.onnx
```

開発用テスト:

```bash
python3 -m unittest discover -s tests
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

- `src/audio_capture/`: WASAPI loopback capture
- `src/dsp/`: EQ、limiter、loudness normalization
- `src/inference/`: ONNX Runtime QNN integration
- `src/profile/`: ローカル個人化プロファイル
- `tests/`: WAV 入出力による DSP の自動テスト
