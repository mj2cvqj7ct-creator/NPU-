# Snapdragon X NPU Audio Enhancer

ARM64 Snapdragon X 搭載 PC の NPU を使い、Spotify、Apple Music、YouTube Music などの再生音を OS レベルで後処理して音質を改善するための設計メモです。

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
5. 各サービスの音量差、圧縮感、帯域バランスの違いを吸収し、同じヘッドホンで一貫した聴感を提供する。

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

NPU には音声波形をそのまま丸投げせず、DSP で抽出した特徴と短い時間窓のスペクトルを入力します。推論結果は「補正パラメータ」として返し、最終的なゲイン変更、EQ、limiter は決定論的な DSP 側で実行します。これにより、NPU 推論の揺らぎがクリックノイズや過大ゲインに直結しない構成にします。

推論タスクは次の 3 系統に分けます。

| タスク | 入力 | 出力 | 目的 |
| --- | --- | --- | --- |
| Content Analyzer | mel spectrogram、短時間 loudness、crest factor | ジャンル傾向、密度、ボーカル優先度 | 曲ごとの補正方針を決める |
| Enhancement Controller | Analyzer 出力、音量、ヘッドホン profile | EQ / compressor / stereo width の目標値 | 派手すぎないリアルタイム補正 |
| Preference Adapter | ローカル操作履歴、プリセット選択、スキップ傾向 | ユーザー嗜好ベクトル | サービス横断の音作り個人化 |

モデルは INT8 量子化を第一候補にし、品質劣化が目立つ帯域推定だけ FP16 を検討します。推論頻度は音声フレームごとではなく 100 ms から 500 ms 間隔に抑え、DSP パラメータをスムージングして低遅延と安定性を両立します。

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
- サービス別の平均ラウドネス、低域量、高域刺激感を推定し、曲単位ではなく短期履歴で補正
- ボーカル、低域、シンバル帯域の過補正を検出して自動的に補正量を下げる

### Phase 3: 個人化

- ユーザーが選んだプリセット、スキップ傾向、音量傾向をローカルで特徴化
- サービス横断で「好みの音作り」を学習
- 推薦サービスの内部ランキングは変更せず、ローカル体験のみを最適化
- 個人化プロファイルは端末内で暗号化し、サービス名、曲名、アーティスト名を保存しない

## 劇的な改善を狙う重点機能

1. **Adaptive Loudness Match**  
   Spotify、Apple Music、YouTube Music 間の体感音量差を短期履歴でならし、曲開始時の急な音量差を抑えます。EBU R128 の integrated loudness だけでなく、低音量再生時の Fletcher-Munson 的な聴感差も補正します。

2. **Neural Tone Shaper**  
   NPU で楽曲の密度、ボーカル存在感、低域過多、高域刺激感を推定し、DSP の dynamic EQ に安全な目標値を渡します。これにより、固定 EQ より曲ごとの破綻が少ない補正を目指します。

3. **Transient Guard / Restore**  
   圧縮感が強い音源ではアタックをわずかに強調し、すでに十分ダイナミックな音源では処理を弱めます。復元風の処理であって原音を再生成するものではないため、true peak limiter と連動させます。

4. **Headphone-Aware Rendering**  
   ヘッドホン別の周波数特性プリセットと、ユーザーの聴取音量に応じた補正量を組み合わせます。Bluetooth 接続時は codec latency と音質劣化を推定し、過度な stereo widening を避けます。

5. **Local Taste Profile**  
   ユーザーが「明瞭」「低音控えめ」「ライブ感」などのプリセットを選んだ結果をローカル特徴量として蓄積し、サービスをまたいで音作りだけを最適化します。推薦順位や配信サービスのデータには介入しません。

## サービス別の扱い

| サービス | 直接改変 | 実現可能な改善 |
| --- | --- | --- |
| Spotify | 不可 | OS 出力音声のリアルタイム補正、ローカル嗜好プロファイル |
| Apple Music | 不可 | ロスレス出力への後処理、ヘッドホン別補正 |
| YouTube Music | 不可 | ブラウザまたはアプリ出力の後処理、音量差補正 |

サービス識別は必須機能ではありません。WASAPI session metadata からアプリ単位で識別できる場合のみ、サービス別の初期値を切り替えます。識別できない場合でも、音声特徴量から補正方針を決めるため、ブラウザ再生や PWA でも基本機能は動作します。

## リアルタイム制約

- 内部処理は 48 kHz / 32-bit float stereo を基準にし、10 ms から 20 ms のオーディオブロックで処理します。
- エンドツーエンド遅延は 40 ms 未満を目標にし、音楽視聴では 60 ms を上限警告にします。
- NPU 推論は別スレッドで先読みし、DSP は前回の安定したパラメータで継続できるようにします。
- 推論がタイムアウトした場合は最後の安全な補正値を保持し、急な bypass による音量変化を避けます。
- すべてのゲイン変更は attack/release smoothing を通し、クリック、ポップ、ポンピングを評価項目に含めます。

詳細なアーキテクチャとモデル仕様は [`docs/snapdragon-x-npu-audio-architecture.md`](docs/snapdragon-x-npu-audio-architecture.md) に分離します。

## 実装ロードマップ

1. WASAPI loopback の最小プロトタイプを作る。
2. 48 kHz stereo のリングバッファと低遅延 DSP チェーンを実装する。
3. ルールベースのラウドネス補正、EQ、limiter を追加する。
4. WAV 入出力のオフライン処理 CLI を作り、DSP 品質と回帰テストを先に固定する。
5. ONNX Runtime QNN Execution Provider で ARM64 / Snapdragon X NPU 推論を試す。
6. Analyzer / Enhancement Controller / Preference Adapter の 3 モデル境界を実装する。
7. NPU が使えない環境では DirectML または CPU fallback に切り替える。
8. ローカル個人化プロファイルを暗号化保存する。
9. APO 化または仮想オーディオデバイス化して常用できる形にする。

## 評価指標

- エンドツーエンド遅延: 40 ms 未満を目標
- クリック、ポップノイズ、ドロップアウトなし
- true peak が 0 dBFS を超えない
- CPU 使用率を低く保ち、NPU 使用時のバッテリー影響を測定
- AB テストで補正オン/オフの主観評価を収集
- LUFS、true peak、短時間 loudness range、spectral centroid、crest factor を自動計測
- NPU 使用時、DirectML fallback、CPU fallback の音質差と遅延差を同じ WAV fixture で比較

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
