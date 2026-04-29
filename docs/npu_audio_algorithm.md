# Snapdragon X NPU 音質改善アルゴリズム設計

この文書は、Spotify、Apple Music、YouTube Music の出力音声を OS レベルで後処理し、Snapdragon X 搭載 ARM64 PC の NPU を使って低遅延に音質を改善するための実装設計です。各サービスのアプリ、DRM、暗号化ストリーム、推薦ランキングは改変しません。

## 実現方針

1. WASAPI loopback または将来の APO で、再生デバイスへ送られる PCM を取得する。
2. CPU/NEON で決定的な DSP を行い、ラウドネス、ピーク、スペクトル特徴を測定する。
3. 10 ms から 20 ms のフレーム単位で NPU へ特徴量を渡し、補正係数または小さな residual 信号を推論する。
4. CPU 側で安全な後処理、true peak limiter、dry/wet 制御を適用し、音割れや過補正を防ぐ。
5. 聴取履歴の特徴とユーザー操作をローカル保存し、サービス横断の好みを「音作り」にだけ反映する。

## リアルタイム処理グラフ

```text
WASAPI loopback PCM
  -> Frame Slicer (48 kHz, stereo, float32, 10-20 ms)
  -> DSP Analyzer
       - momentary loudness
       - true peak estimate
       - spectral centroid / tilt
       - vocal-band energy
       - low-band density
       - clipping and intersample risk
  -> Feature Normalizer
  -> NPU Model Selector
       - QNN Execution Provider on Snapdragon X
       - DirectML fallback
       - CPU fallback
  -> NPU Inference
       - content embedding
       - enhancement coefficients
       - optional residual enhancement
  -> Safety DSP
       - dynamic EQ
       - multiband compression
       - transient guard
       - stereo width clamp
       - true peak limiter
  -> Render PCM
```

## NPU に載せる処理

NPU は、固定式 DSP よりも内容依存の判断が必要な軽量推論に限定します。ピークリミッターやクリッピング保護のような安全制約は、推論結果に依存させず CPU 側で必ず実行します。

### 入力特徴量

| 名前 | 形状 | 説明 |
| --- | --- | --- |
| `log_mel` | `[1, 64, 20]` | 20 フレーム分の log-mel スペクトログラム |
| `loudness_stats` | `[1, 6]` | momentary/short-term loudness、crest factor、true peak など |
| `service_hint` | `[1, 3]` | Spotify、Apple Music、YouTube Music を推定できた場合の one-hot。未確定時はゼロ |
| `device_profile` | `[1, 16]` | ヘッドホン EQ、出力デバイス、音量帯の埋め込み |
| `preference_vector` | `[1, 16]` | ローカル嗜好プロファイル。外部送信しない |

### 出力

| 名前 | 形状 | 安全制約 |
| --- | --- | --- |
| `eq_gains_db` | `[1, 8]` | 各バンドの変化量を -4 dB から +4 dB に制限 |
| `compression_curve` | `[1, 6]` | attack/release/ratio/knee を許容範囲に clamp |
| `clarity_mix` | `[1, 1]` | ボーカル明瞭化の dry/wet。最大 0.35 |
| `transient_mix` | `[1, 1]` | トランジェント復元風補正。最大 0.25 |
| `stereo_width` | `[1, 1]` | 0.85 から 1.15 に制限し、モノ互換性を保つ |

Residual 信号を出すモデルは Phase 2 後半以降の候補です。初期段階では補正係数だけを NPU から返し、DSP 側で処理する方が安全で検証しやすくなります。

## DSP アルゴリズム

### 1. ラウドネス安定化

- EBU R128 ベースの short-term loudness を計算する。
- サービスや楽曲ごとの差を小さくするため、目標値は初期値 -16 LUFS にする。
- ゲイン変更は 1 秒あたり 1.5 dB 以下に制限し、ポンピングを避ける。
- Apple Music のロスレス再生では補正を控えめにし、YouTube Music では動画由来の音量差に強めに反応する。

### 2. 動的 EQ

8 バンドの低遅延 IIR EQ を基本にします。

| Band | 中心周波数 | 主な用途 |
| --- | --- | --- |
| 1 | 60 Hz | 低域の過不足補正 |
| 2 | 120 Hz | ベース密度の調整 |
| 3 | 250 Hz | こもり抑制 |
| 4 | 500 Hz | 中低域の濁り調整 |
| 5 | 1.5 kHz | ボーカル存在感 |
| 6 | 3 kHz | 明瞭度 |
| 7 | 6 kHz | アタック、歯擦音管理 |
| 8 | 12 kHz | 空気感 |

NPU はバンドごとの目標ゲインを提案し、DSP はヘッドホン補正、ユーザー設定、true peak 余裕を加味して最終値を決めます。

### 3. マルチバンドコンプレッション

- 低域、中域、高域の 3 バンドに分ける。
- 低域は過大なベースだけを抑え、常時圧縮しない。
- 中域はボーカル明瞭度を支えるため、強いコンプレッションを避ける。
- 高域は歯擦音検出時に一時的に de-ess する。

### 4. トランジェント保護

圧縮音源を派手に補正するとアタックが潰れやすいため、短時間エネルギー差分でトランジェントを検出し、その区間では EQ とコンプレッションの変化量を抑えます。

### 5. true peak limiter

- 出力 true peak は -1.0 dBTP 以下を目標にする。
- limiter は最後段に置き、NPU の出力より優先する。
- 連続的に limiter が動作する場合は、前段の loudness gain と EQ boost を自動的に下げる。

## サービス別プロファイル

| サービス | 課題 | 初期補正方針 |
| --- | --- | --- |
| Spotify | 圧縮音源、音量正規化設定、端末間差 | トランジェント保護と中高域の明瞭度を控えめに追加 |
| Apple Music | ロスレス/Atmos/通常 AAC が混在 | ロスレス時は低侵襲、音場を広げすぎない |
| YouTube Music | 動画由来の音量差、アップロード音源の品質差 | ラウドネス安定化とピーク保護を強めにする |

サービス判定はプロセス名、セッション名、ユーザー指定のどれかで行います。判定できない場合は `generic_streaming` として扱い、最も保守的な補正を使います。

## 個人化

個人化は「どの楽曲を推薦するか」ではなく「どう鳴らすと聴きやすいか」に限定します。

- ユーザーが選んだプリセット、音量帯、手動 EQ 変更をローカル特徴として保存する。
- スキップや再生停止は、音質補正への反応としてのみ集計し、楽曲 ID やアカウント情報は保存しない。
- profile vector はデバイス内で生成し、外部 API へ送信しない。
- プロファイルはエクスポート可能にする場合でも、既定では暗号化保存する。

## NPU ランタイム選択

1. ONNX Runtime QNN Execution Provider を優先する。
2. QNN 初期化に失敗した場合は DirectML へ切り替える。
3. どちらも使えない場合は CPU fallback を使い、NPU 専用の residual enhancer を無効化する。

初期化時に一度だけバックエンドを決定し、再生中に頻繁に切り替えないようにします。再生中のバックエンド変更が必要な場合は、dry/wet を 0 にフェードしてからモデルを差し替えます。

## 品質ゲート

- 48 kHz stereo で 20 ms フレーム処理時、処理時間は 5 ms 未満を目標にする。
- エンドツーエンド遅延は 40 ms 未満を維持する。
- 補正オン/オフの AB テストで、音量差だけによる有利不利が出ないよう loudness match する。
- すべての出力サンプルは `[-1.0, 1.0]` に収める。
- true peak limiter の平均動作量が高い場合は、改善ではなく過補正として扱う。

## 実装順序

1. JSON プロファイルからサービス別の補正上限を読み込む。
2. WAV 入出力のオフライン DSP テストで loudness、EQ、limiter を検証する。
3. WASAPI loopback へ接続し、generic profile でリアルタイム処理する。
4. ONNX の coefficient model を接続し、QNN/DirectML/CPU の順で backend selection を実装する。
5. ローカル個人化 profile vector を追加し、補正量の微調整だけに使う。
