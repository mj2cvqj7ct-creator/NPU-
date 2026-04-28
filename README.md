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
python3 guardian_blacklist.py watch-log ./security.log --threshold 5
python3 guardian_blacklist.py report ./incident_report.md
```

## 起動時に自動開始する

Linuxでは、ログイン時に常駐監視を開始するsystemd user serviceを生成できます。

```bash
python3 guardian_blacklist.py install-autostart ./security.log --threshold 5
systemctl --user enable --now guardian-blacklist.service
```

`install-autostart --enable` を指定すると、サービスファイル生成後に `systemctl --user enable --now` まで実行します。実際にローカルファイアウォールへ適用する場合は `--apply` を追加してください。管理者権限やOS側のファイアウォール設定が必要になることがあります。

## 保存先

既定では `~/.local/share/guardian-blacklist/blacklist.json` に保存します。テストや別環境では `GUARDIAN_BLACKLIST_HOME` または `--data-dir` で変更できます。

## 安全上の制限

- ブロック対象はグローバル公開IPv4のみです。
- プライベート、ループバック、予約済み、マルチキャストIPは拒否します。
- 外部機関への送信や登録は行いません。
- レポートは手動確認・手動提出用です。
- 常駐監視はローカルログの解析とローカルブラックリスト登録だけを行います。
