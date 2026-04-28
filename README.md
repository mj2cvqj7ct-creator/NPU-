# NPU Audio Enhancer

ARM64 Snapdragon X 搭載 PC の NPU を使い、Spotify、Apple Music、YouTube Music などの再生音をシステム全体で後処理するための土台です。

## 重要な前提

- Spotify、Apple Music、YouTube Music の暗号化された音源や公式推薦アルゴリズムを直接改変することはできません。
- 実装可能な範囲は、OS の音声出力後に動くポストプロセッサ、またはエクスポート可能な再生履歴・プレイリストを使った推薦補助です。
- Snapdragon X の NPU 実行は Windows ARM64 上で ONNX Runtime QNN Execution Provider などに接続する想定です。このリポジトリの CLI は同じ処理方針を CPU で検証できます。

## できること

- デスクトップアプリは、Spotify、Apple Music、YouTube Music のリアルタイム再生音を処理するための NPU コントロール画面です。
- Snapdragon X NPU を ONNX Runtime QNN Execution Provider で使う構成を前提にしています。
- ASIO の SABAJ A20D(ES) / XMOS USB DAC Driver Control Panel を低レイテンシ出力経路として想定し、極小バッファ、排他出力、NPU 直後の最短出力を画面に表示します。
- `holographic-vocal-stage` プロファイルで、定位、ホログラフィックな奥行き、楽器分離、ボーカルの前景化を狙います。
- WAV 検証用パイプラインは、ピーク正規化、帯域別フォーカス、ボーカル帯域強調、mid/side ステレオ拡張、ソフトクリップを実行します。
- Deep Learning 型のローカル推薦 AI が、Spotify、Apple Music、YouTube Music の履歴/プレイリスト信号を埋め込み化し、リアルタイムの次候補キューや同期プレイリスト案として反映します。
- WAV 音声に対して、ピーク正規化、軽いダイナミックレンジ圧縮、ソフトクリップを適用するローカル検証用 CLI もあります。
- CLI は `--profile snapdragon-x-npu` で、Snapdragon X NPU ターゲットのプロファイルを選べます。
- ローカル検証用のデモ WAV を生成できます。
- 公式ストリーミングアプリ内の暗号化音源やサーバー側推薦アルゴリズムは直接改変せず、OS ミキサー後のリアルタイム音声後処理と、履歴/プレイリスト由来のローカル推薦補助で強化します。

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
python3 -m npu_audio_enhancer demo_input.wav demo_enhanced.wav --profile holographic-vocal-stage
npu-audio-enhancer-gui
```

## Windows デスクトップショートカット

Windows 実機のデスクトップにネイティブ EXE のショートカットを作る場合は、PowerShell で次を実行します。

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\windows\create_desktop_shortcut.ps1
```

## Windows ネイティブ EXE

Python や PowerShell なしで起動できる Windows デスクトップアプリは、`dist/windows/NPUStreamingMusicEnhancer.exe` に生成されます。再ビルドする場合は次を実行します。

```bash
bash native/windows/build.sh
```

同じビルドで `dist/windows/NPUStreamingMusicEnhancerInstaller.exe` も生成されます。Windows でこのインストーラーを実行すると、アプリを `%LOCALAPPDATA%\NPUStreamingMusicEnhancer\` にコピーし、Windows デスクトップにショートカットを作成します。

Windows ARM64 実機で Snapdragon X NPU を使うには、QNN 対応 ONNX Runtime、NPU ドライバー、ASIO 対応 XMOS/SABAJ DAC ドライバーを導入し、`docs/snapdragon_x_plan.md` の接続箇所へ実機 I/O を実装してください。このリポジトリではクラウド CI で動く CPU 検証パスと、Windows EXE の UI/状態制御を提供します。

## テスト

```bash
python3 -m unittest discover -s tests
```

## 次に実機でつなぐ場所

詳しい設計は `docs/snapdragon_x_plan.md` を参照してください。
