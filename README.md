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
