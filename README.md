
# binance-breakpoint — Martingale + Trail-Time Backtester (Docker, GitHub-ready)

> Core idea: **สะสมซื้อแบบบันได (Martingale by distance + amount) + ตั้งพื้นตามกำไร (Trail Floor) + เพดานเวลา**  
> มี **โครง (scaffold) สำหรับ `Buy the Dip (BTD)` / `Sell at the Height (SAH)`** ปิดไว้โดยดีฟอลต์ (รอคอนเฟิร์มค่า)

## Quick Start (Docker)
```bash
mkdir -p out data
cp .env.example .env    # ใส่ API KEY/SECRET ของ Binance ที่นี่ (อย่า commit)
docker compose build
docker compose build backtest demo live
# Live (ดึงจาก Binance ตาม config.yaml)
docker compose up backtest
# Demo (ออฟไลน์ - จำลองราคาเพื่อดู flow)
docker compose up demo
```

## New: lightweight HTTP backtest service (port 8181)
Run an isolated backtest API + minimal UI (separateจาก live HTML):

```bash
docker compose build backtest
docker compose up backtest  # serves http://localhost:8181
```

- Endpoint: `POST /backtest` with body `{ "usdt": 1000, "symbols": ["BTCUSDT"], "intervals": ["1d","7d",...], "top_n": [30,100,500] }`
- Defaults: auto-discovers liquid USDT pairs (top 50%), tests intervals `1d,7d,30d,1m,3m,6m,1y`, returns ranked `top30/top100/top500` by aggregated PnL.
- UI: open `http://localhost:8181/` for a single-page form, loading indicator, result tables, and CSV/JSON download links for `top100` and `top500`.
- The service is read-only and does not affect the live service on port 8080.

ผลลัพธ์: `./out/<SYMBOL>_{events,equity,trades,summary}.(csv/json)` + `plot.png`

### Persist live savepoints (Windows/Mac/Linux)
- ดีฟอลต์: `live` จะเขียนไฟล์สถานะไปที่ `/app/savepoint` ในคอนเทนเนอร์ และ **bind mount** ไปยังโฟลเดอร์ `./savepoint` ที่อยู่ข้างไฟล์ `docker-compose.yml` (บน Windows จะเห็นที่ `C:\\\\...\\binance-breakpoint\\savepoint`).
- สามารถย้ายที่เก็บได้ด้วยตัวเลือก `--savepoint-dir` หรือกำหนด env `SAVEPOINT_DIR` (เช่น `SAVEPOINT_DIR=/app/savepoint`) แล้ว mount path นั้นจากโฮสต์
  ```yaml
  # docker-compose.yml (ตัวอย่าง)
  services:
    live:
      environment:
        - SAVEPOINT_DIR=/app/savepoint
      volumes:
        - ./savepoint:/app/savepoint   # บน Windows ใช้เส้นทางเดียวกันนี้ได้
  ```
- ตรวจสอบว่าโฟลเดอร์ `savepoint` ถูกสร้างบนโฮสต์ก่อนรัน (หรือปล่อยให้ Docker สร้างให้) เพื่อให้สถานะไม่หายเวลา restart/สร้างคอนเทนเนอร์ใหม่

- ต้องการทดสอบว่าไฟล์บันทึกถูกสร้างจริงหรือไม่ (dry-run ก็ได้):
  ```bash
  mkdir -p savepoint/test
  python live_trader.py --symbol ZECUSDT --dry-run --poll-seconds 2 --savepoint-dir savepoint/test
  # รอให้รัน 2-3 รอบแล้วกด Ctrl+C
  ls savepoint/test
  python summary.py --savepoint-dir savepoint/test --out-dir out
  ```
  สคริปต์จะสร้าง `ZECUSDT.json` พร้อมไฟล์ event log (`*_event_log.{jsonl,csv}`) และสรุป PnL ที่ดึงจากไฟล์บันทึกได้ ใช้ขั้นตอนเดียวกันนี้กับ Docker (`docker compose up live`) โดยตรวจสอบไฟล์ในโฟลเดอร์ `savepoint`

### PnL Summary (Backtest + Live)
ใช้ `summary.py` เพื่อรวบรวม PnL จากไฟล์ backtest (`out/*summary*`) และสถานะ live (`savepoint/<SYMBOL>.json`)
```bash
python summary.py --savepoint-dir savepoint --out-dir out
```
สคริปต์จะแสดงตาราง PnL รายไฟล์ + รวมตามคู่ แยกผลจากราคาเคลื่อนไหว (unrealized) กับกำไรขาดทุนจากเทรดที่ปิดไป (realized)


## Config
- `config.yaml` — ตั้งค่าคู่/ช่วง/ฟีส์/พารามิเตอร์หลัก และ **features.BTD/SAH** (disabled)
  - เพิ่ม `features.scalp_mode` เพื่อ "เข้าเร็ว-ออกเร็ว" 1-3 ไม้แรก: กำหนด % ย่อที่ให้ซื้อ, % เด้งที่ขายทำกำไร และให้สเกลตามช่วงแกว่งของวัน (วัดจาก high-low ของรอบ) โดยจะหยุดหลังครบ `max_trades` และไม่ไปกินทุนบันไดหลัก (`b_alloc`) เกินโควตา `order_pct_allocation`
  - โครง `features.buy_the_dip` / `features.sell_at_height` ยังเป็นการ "ตั้งธง" เฉย ๆ (log event) ไม่ได้ยิงออเดอร์จริง เพื่อให้เห็นว่าเงื่อนไขถึงจุดหรือยังก่อนปรับค่าใช้งาน
  - บันไดซื้อเลือกสเกลขนาดแบบ `size_mode: geometric|fibonacci`, จัดระยะห่างแบบ `gap_mode: additive|multiplicative (gap_factor)` และตั้งเพดาน `max_quote_per_leg`/`max_total_quote` เพื่อลดการทบเร็วเกินไป
  - `general.profit_recycling` (เปิด/ปิดได้) จะหักกำไร 10% ของแต่ละรอบไว้ซื้อ coin ของคู่เทรด (ใช้ทุนสะสมขั้นต่ำ 10 USDT) เพื่อรอขายตอน rip ตามสัญญาณ SAH และรวบกำไรจากทุกคู่ 5% ไปซื้อ BNB เพื่อลดค่าฟี; รายการอัปเดตถูกเก็บในไฟล์แยก `profit_accumulation.json` ใต้ savepoint
- `.env` — ใส่ `BINANCE_API_KEY/BINANCE_API_SECRET` (ไม่ใช้กับ klines public; เตรียมไว้สำหรับงานถัดไป)

### Live Trader
- `live_trader.py` จะอ่าน `config.yaml` และรันกลยุทธ์แบบรอบต่อรอบกับแท่งราคาที่ปิดล่าสุด (`klines`) จาก Binance
- เมื่อเกิดสัญญาณซื้อ/ขายในบันไดหลัก จะส่งคำสั่ง **MARKET** ที่ปริมาณเท่ากับบันไดนั้น (BUY) หรือปริมาณคงเหลือ (SELL)
- ตั้งค่าตัวเลือก `general.sell_scale_out` เพื่อ "เฉลี่ยขาย" ตอนมีกำไร: กำหนด `chunks` (จำนวนไม้ย่อย) และ `delay_seconds` (พักระหว่างไม้) ถ้าเปิด `profit_only: true` จะเฉลี่ยขายเฉพาะตอนสัญญาณทำกำไรเท่านั้น
- ระบบจะเช็คยอด `quote` ฟรีก่อนยิงคำสั่ง BUY และจะลดขนาดออเดอร์ให้อัตโนมัติถ้าเงินคงเหลือไม่พอ เพื่อกัน error `Account has insufficient balance` แล้วเลื่อนขั้นบันไดผิดพลาด
- ตั้ง `--dry-run` เพื่อทดลองโดยไม่ส่งคำสั่งจริง: `python live_trader.py --dry-run --symbol ZECUSDT`
- ใน Docker ใช้บริการ `live` (ต้องมี `.env` ใส่คีย์) และควรติดตาม log อย่างใกล้ชิด

- เปิด HTTP control server ดีฟอลต์ที่พอร์ต `8080` สำหรับเช็ค health และสั่งหยุด/ทำงานต่อ (ตั้ง `--http-port 0` เพื่อปิด)
  - เช็คสถานะ: `curl http://localhost:8080/health`
  - หยุดทุกคู่: `curl -X POST http://localhost:8080/pause`
  - หยุดเฉพาะคู่: `curl -X POST 'http://localhost:8080/pause?symbol=ZECUSDT'`
  - กลับมาทำงาน: `curl -X POST http://localhost:8080/resume`

> **สำคัญ**: ห้าม commit คีย์จริงขึ้น GitHub (ไฟล์ `.env` ถูก ignore แล้ว)
