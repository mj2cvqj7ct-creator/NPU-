# Guardian Blacklist

自分のPCを守るためのローカル防御CLIです。不審な公開IPv4アドレスを記録し、OS別のファイアウォールブロックコマンドを表示し、プロバイダ・警察・銀行などへ手動相談するための証跡レポートを生成します。

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
python3 guardian_blacklist.py report ./incident_report.md
```

## 保存先

既定では `~/.local/share/guardian-blacklist/blacklist.json` に保存します。テストや別環境では `GUARDIAN_BLACKLIST_HOME` または `--data-dir` で変更できます。

## 安全上の制限

- ブロック対象はグローバル公開IPv4のみです。
- プライベート、ループバック、予約済み、マルチキャストIPは拒否します。
- 外部機関への送信や登録は行いません。
- レポートは手動確認・手動提出用です。

## Windows LDAC Assistant

Windows 標準 Bluetooth スタックには LDAC エンコーダーが含まれていないため、この補助アプリは独自コーデックの生成、ドライバー改変、OS制限の回避は行いません。代わりに、LDAC 利用可否の診断、暗号化された希望設定の保存、ユーザーログオン時の自動起動登録を行います。

```bash
python3 windows_ldac_assistant.py status --system Windows
python3 windows_ldac_assistant.py configure --preferred-bitrate 990 --start-on-login --dry-run
python3 windows_ldac_assistant.py monitor
```

Windows では設定を現在の Windows ユーザーに紐づく DPAPI で保護し、`--start-on-login` を指定すると `HKCU\Software\Microsoft\Windows\CurrentVersion\Run` に監視コマンドを登録します。実際に登録する場合は Windows 上で `--dry-run` を外して実行してください。
