## Raspberry Pi Discord ステータスボット

Raspberry Pi の CPU/メモリ使用率と CPU/GPU 温度を 5 秒間隔で収集し、30 秒ごとの推移を蓄積した上で、日本時間の毎時00分に 30 秒グラフと 1 時間グラフをまとめて Discord チャンネルへ投稿する py-cord 製ボットです。

### 主な機能
- `psutil` と Raspberry Pi の温度ファイルからメトリクスを取得。
- 直近 30 秒分と直近 1 時間分の推移を、両方とも日本時間で毎時ちょうど（分00）にまとめて投稿（30 秒ごとには送信しません）。投稿時は最新状態の Embed とグラフ画像2枚をセットで送信。
- matplotlib で CPU/メモリ使用率、CPU/GPU 温度、ネットワーク送受信速度、Ping 平均値・パケットロス率、サーマルスロットリング有無をまとめた 4 段グラフを生成。
- Slash Command `/raspi_status` で、現在の状態を即座に確認可能（Embed 表示）。
- `.env` に書いたトークン情報を自動で読み込むため、端末に環境変数を永久設定する必要なし。

### 動作条件
- Python 3.12.11 で動作する Raspberry Pi OS (vcgencmd 利用のため `libraspberrypi-bin` 推奨)。
- Discord サーバーに参加済みの Bot ユーザーと投稿対象のテキストチャンネル。
- 依存パッケージ: `py-cord`, `psutil`, `matplotlib`（`requirements.txt` 参照）。
- `ping` コマンドが利用可能であること（ネットワーク品質計測に使用）。

### セットアップ手順
1. **Discord Bot の作成**
   - [Discord Developer Portal](https://discord.com/developers/applications) でアプリケーションを作り、Bot を作成してサーバーへ招待。
   - Bot のトークンを控え、必要に応じてメッセージ送信/ファイル添付権限を付与。

2. **Python 環境の準備**
   ```bash
   sudo apt update
   sudo apt install -y python3.12 python3.12-venv python3-pip libraspberrypi-bin
   python3.12 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **`.env` の作成（必須）**
   - `.env.example` をコピーし、Discord Bot のトークンと投稿先チャンネル ID を入力します。
     ```bash
     cp .env.example .env
     nano .env  # もしくは好みのエディタで編集
     ```
   - `.env` は `.gitignore` 登録済みのため、GitHub へトークンがアップロードされる心配はありません。
   - **Slash Command を確実に表示させるため、必ず `DISCORD_GUILD_ID` にサーバー ID を設定してください。**
     - グローバルコマンドは反映まで最大 1 時間かかるため、ギルドコマンドで同期する仕様にしています。設定が無いと起動エラーになります。

### 実行方法
```bash
source .venv/bin/activate
python bot.py
```
起動直後から 5 秒ごとにメトリクスを蓄積し、十分なデータが揃い次第、毎時00分に 30 秒グラフと 1 時間グラフを最新状態の Embed とともにまとめて投稿します。

### カスタマイズ
- 収集間隔や投稿間隔を変更する場合は `bot.py` 内の `SAMPLE_INTERVAL_SECONDS`, `SHORT_WINDOW`, `LONG_WINDOW` を書き換えてください。
- `render_graph` 関数を編集するとグラフの構成やデザインを細かく調整できます。
- ネットワーク品質計測の宛先や試行回数を変える場合は `PING_TARGET`, `PING_COUNT`, `PING_TIMEOUT_SECONDS` を編集してください。
- Discord 上で即時確認したい場合は `/raspi_status` コマンドを実行すると最新メトリクスが Embed で返ります。

### トラブルシューティング
- GPU 温度が常に空になる場合は `vcgencmd measure_temp` が実行できるか確認してください。
- Bot がチャンネル ID を解決できない場合は、Bot がそのサーバーに参加しているか、開発者モードで取得した ID に誤りがないか見直してください。
- 画面のない環境でも描画できるよう `Agg` バックエンドを使用しています。別の環境で GUI 出力が必要な場合は `matplotlib.use()` を調整してください。
