
# binance-breakpoint — Martingale + Trail-Time Backtester (Docker, GitHub-ready)

> Core idea: **สะสมซื้อแบบบันได (Martingale by distance + amount) + ตั้งพื้นตามกำไร (Trail Floor) + เพดานเวลา**  
> มี **โครง (scaffold) สำหรับ `Buy the Dip (BTD)` / `Sell at the Height (SAH)`** ปิดไว้โดยดีฟอลต์ (รอคอนเฟิร์มค่า)

## Quick Start (Docker)
```bash
mkdir -p out data
cp .env.example .env    # ใส่ API KEY/SECRET ของ Binance ที่นี่ (อย่า commit)
docker compose build
docker compose build backtest demo
# Live (ดึงจาก Binance ตาม config.yaml)
docker compose up backtest
# Demo (ออฟไลน์ - จำลองราคาเพื่อดู flow)
docker compose up demo
```

ผลลัพธ์: `./out/<SYMBOL>_{events,equity,trades,summary}.(csv/json)` + `plot.png`

## Config
- `config.yaml` — ตั้งค่าคู่/ช่วง/ฟีส์/พารามิเตอร์หลัก และ **features.BTD/SAH** (disabled)
- `.env` — ใส่ `BINANCE_API_KEY/BINANCE_API_SECRET` (ไม่ใช้กับ klines public; เตรียมไว้สำหรับงานถัดไป)

> **สำคัญ**: ห้าม commit คีย์จริงขึ้น GitHub (ไฟล์ `.env` ถูก ignore แล้ว)
