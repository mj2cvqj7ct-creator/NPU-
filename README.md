# Guardian Blacklist

自分のPCを守るためのローカル防御CLIです。不審な公開IPv4/IPv6アドレスを記録し、OS別のファイアウォールブロックコマンドを表示し、プロバイダ・警察・銀行などへ手動相談するための証跡レポートを生成します。

このアプリは法律違反や誤通報を避けるため、第三者機関への自動登録、電話・手紙・メッセージ送信、プロバイダへの停止要求は行いません。

## 使い方

```bash
python3 guardian_blacklist.py add 8.8.8.8 \
  --reason "不審な接続試行" \
  --source "firewall.log" \
  --evidence "ログに短時間で複数回出現"
```

実際にローカルファイアウォールへ適用する場合は、表示されたコマンドを確認してから管理者権限で実行してください。`--apply` を付けるとアプリが直接実行します。

```bash
python3 guardian_blacklist.py list
python3 guardian_blacklist.py scan-log ./security.log --threshold 5
python3 guardian_blacklist.py watch-log ./security.log --threshold 5
python3 guardian_blacklist.py scan-log ./security.log --port-scan-threshold 10
python3 guardian_blacklist.py report ./incident_report.md
python3 guardian_blacklist.py report ./international_report.md --audience international
python3 guardian_blacklist.py report ./japan_international_report.md --audience japan-international
python3 guardian_blacklist.py abuseipdb-report ./abuseipdb_manual.json
python3 guardian_blacklist.py threat-intel-report ./weekly_exports
python3 guardian_blacklist.py watch-log ./security.log --abuseipdb-export ./abuseipdb_manual.json
```

## Cursorなしで起動時に自動開始する

Linuxでは、OS起動時に常駐監視を開始するsystemd system serviceを生成できます。`--apply` を付けると、新しく見つかった公開IPv4/IPv6をプロトコルを限定せずローカルファイアウォールへ適用します。ポートスキャン兆候は、同じIPから観測された異なる宛先ポート数が `--port-scan-threshold` 以上になった場合に記録します。ログに `proto=TCP` などが含まれる場合は証跡へ残します。`--abuseipdb-export` を付けると、検知のたびにAbuseIPDB手動提出用ファイルをローカル更新します。

```bash
sudo python3 guardian_blacklist.py install-boot-service /var/log/security.log --threshold 5 --port-scan-threshold 10 --abuseipdb-export /var/lib/guardian-blacklist/abuseipdb_manual.json --apply --enable
```

ログイン時だけ常駐させたい場合は、systemd user serviceも生成できます。

```bash
python3 guardian_blacklist.py install-autostart ./security.log --threshold 5 --enable
```

既定の監視間隔は1秒です。より間隔を空けたい場合は `--interval` を指定してください。管理者権限やOS側のファイアウォール設定が必要になることがあります。

## 週次で提出候補ファイルを生成する

AlienVault OTX、IBM X-Force、AbuseIPDB向けに、週1回の手動提出用ファイル生成timerを作れます。外部API送信は行いません。

```bash
python3 guardian_blacklist.py install-weekly-export ./weekly_exports --enable
```

## 保存先

既定では `~/.local/share/guardian-blacklist/blacklist.json` に保存します。テストや別環境では `GUARDIAN_BLACKLIST_HOME` または `--data-dir` で変更できます。

## 安全上の制限

- ブロック対象はグローバル公開IPv4/IPv6のみです。
- プライベート、ループバック、予約済み、マルチキャストIPは拒否します。
- 外部機関への送信や登録は行いません。
- レポートは手動確認・手動提出用です。
- AbuseIPDB、AlienVault OTX、IBM X-Force向けファイルは自動生成できますが、手動提出用でAPI送信は行いません。
- 常駐監視はローカルログの解析、ポートスキャン/IPスキャン兆候とプロトコル情報の記録、ローカルブラックリスト登録だけを行います。

## Windows LDAC Assistant

Windows 標準 Bluetooth スタックには LDAC エンコーダーが含まれていないため、この補助アプリは独自コーデックの生成、ドライバー改変、OS制限の回避は行いません。代わりに、LDAC 利用可否の診断、暗号化された希望設定の保存、ユーザーログオン時の自動起動登録を行います。

```bash
python3 windows_ldac_assistant.py status --system Windows
python3 windows_ldac_assistant.py configure --preferred-bitrate 990 --start-on-login --dry-run
python3 windows_ldac_assistant.py monitor
```

Windows では設定を現在の Windows ユーザーに紐づく DPAPI で保護し、`--start-on-login` を指定すると `HKCU\Software\Microsoft\Windows\CurrentVersion\Run` に監視コマンドを登録します。実際に登録する場合は Windows 上で `--dry-run` を外して実行してください。

## Audio Lossless Assistant

MP3、AAC、LDAC、SBC などの非可逆コーデックで一度失われた音声情報は、後からどのアプリでも完全復元できません。この補助アプリは「全コーデックをロスレス化」と偽らず、入力コーデックがロスレスか判定し、FLAC、ALAC、WAV/PCM などの実際のロスレス保存先への保全計画を作成します。

```bash
python3 audio_lossless_assistant.py assess mp3
python3 audio_lossless_assistant.py assess flac
python3 audio_lossless_assistant.py plan ldac --target-codec flac
python3 audio_lossless_assistant.py plan wav --target-codec alac --output ./lossless_plan.json
```

非可逆コーデックからロスレス形式へ変換する場合、保存できるのは「デコード後に残っている波形」だけです。元ファイルを必ず保持し、出力には `preserved-from-lossy` のように復元ではないことを明記してください。

## AI/NPU Audio Enhancement Assistant

Bluetooth コーデックで劣化した音声に対して、AI と NPU でノイズ低減、帯域拡張、アーティファクト低減の計画を作成できます。ただし、推定補完であり、失われた元サンプルを証明可能なハイレゾロスレスとして復元するものではありません。

```bash
python3 npu_audio_enhancement_assistant.py status
python3 npu_audio_enhancement_assistant.py plan ldac --target flac-24-96
python3 npu_audio_enhancement_assistant.py plan sbc --target wav-24-96 --output ./npu_enhancement_plan.json
```

ONNX Runtime の NPU 実行プロバイダーが見つかれば NPU 利用計画を表示し、見つからない場合は CPU フォールバックとして表示します。出力には必ず `ai-enhanced-high-res-preservation` のように AI 補完であることを明記してください。

## デスクトップアプリ

Tkinter GUI でロスレス判定、AI/NPU補完計画、Windows LDAC 診断を操作できます。

```bash
python3 audio_desktop_app.py
```

Linux デスクトップ環境では `audio-lossless-assistant.desktop` をデスクトップへコピーして、必要に応じて実行権限を付けてください。
