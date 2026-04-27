# NPU Audio Enhancer

ARM64 Snapdragon X 搭載 PC の NPU を使い、Spotify、Apple Music、YouTube Music などの再生音をシステム全体で後処理するための土台です。

## 重要な前提

- Spotify、Apple Music、YouTube Music の暗号化された音源や公式推薦アルゴリズムを直接改変することはできません。
- 実装可能な範囲は、OS の音声出力後に動くポストプロセッサ、またはエクスポート可能な再生履歴・プレイリストを使った推薦補助です。
- Snapdragon X の NPU 実行は Windows ARM64 上で ONNX Runtime QNN Execution Provider などに接続する想定です。このリポジトリの CLI は同じ処理方針を CPU で検証できます。

## できること

- WAV 音声に対して、ピーク正規化、軽いダイナミックレンジ圧縮、ソフトクリップを適用します。
- `--profile snapdragon-x-npu` で、Snapdragon X NPU ターゲットのプロファイルを選べます。
- ローカル検証用のデモ WAV を生成できます。
- デスクトップアプリからデモ生成、WAV 選択、音質向上処理を実行できます。

## セットアップ

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

Linux で GUI を表示する場合は、Qt の xcb/EGL ランタイムライブラリが必要です。

## 実行例

```bash
python3 -m npu_audio_enhancer --generate-demo demo_input.wav
python3 -m npu_audio_enhancer demo_input.wav demo_enhanced.wav --profile snapdragon-x-npu
npu-audio-enhancer-gui
```

## テスト

```bash
python3 -m unittest discover -s tests
```

## 次に実機でつなぐ場所

詳しい設計は `docs/snapdragon_x_plan.md` を参照してください。
