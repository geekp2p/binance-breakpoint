# Setup Guide (Windows & Ubuntu)

เอกสารนี้อธิบายขั้นตอนสำหรับเตรียมสภาพแวดล้อมเพื่อรัน **binance-breakpoint** ด้วย Docker บน Windows และ Linux (Ubuntu). ครอบคลุมทั้ง Docker image, `docker compose`, และเคล็ดลับการตั้งค่าไฟล์คอนฟิกที่โปรเจกต์ต้องการ

## 1. สิ่งที่ต้องมี (Prerequisites)

| รายการ | Windows | Ubuntu |
|--------|---------|--------|
| Git | [ดาวน์โหลด](https://git-scm.com/download/win) | `sudo apt install git` |
| Docker Engine | ติดตั้งผ่าน Docker Desktop (รวม Docker Compose) | `sudo apt install docker.io docker-compose-plugin` หรือใช้สคริปต์จาก Docker | 
| Docker Compose | มาพร้อม Docker Desktop | ติดตั้งผ่านแพ็กเกจ `docker-compose-plugin` (Compose V2) |
| Python 3.11 (ทางเลือก) | ติดตั้งจาก Microsoft Store หรือ python.org (ใช้เมื่อต้องการรันนอก Docker) | `sudo apt install python3.11 python3.11-venv` |

หมายเหตุ: โปรเจกต์นี้เน้นการใช้งานผ่าน Docker เป็นหลัก ดังนั้นไม่จำเป็นต้องลง Python บนโฮสต์ หากใช้งานผ่านคอนเทนเนอร์ทั้งหมด

## 2. เตรียมซอร์สโค้ด

1. เปิดเทอร์มินัล (PowerShell บน Windows, Terminal บน Ubuntu)
2. โคลนโปรเจกต์
   ```bash
   git clone https://github.com/<your-account>/binance-breakpoint.git
   cd binance-breakpoint
   ```
3. สร้างโฟลเดอร์ผลลัพธ์และข้อมูล
   ```bash
   mkdir -p out data
   ```
4. คัดลอกไฟล์ `.env.example` เป็น `.env`
   ```bash
   cp .env.example .env
   ```
5. แก้ไขไฟล์ `.env` เพื่อใส่ `BINANCE_API_KEY` และ `BINANCE_API_SECRET` (ห้าม commit ไฟล์นี้)

## 3. การใช้งานบน Windows (Docker Desktop + WSL2)

1. ติดตั้ง [Docker Desktop](https://www.docker.com/products/docker-desktop/) และเปิดใช้งาน WSL2 ตามตัวช่วยตั้งค่า (Setup Wizard)
2. เปิด Docker Desktop ให้พร้อมทำงาน
3. เปิด PowerShell/Windows Terminal แล้วไปยังโฟลเดอร์โปรเจกต์ (`cd binance-breakpoint`)
4. สร้าง/ปรับไฟล์คอนฟิกตามข้อ 2 ด้านบน
5. สร้างอิมเมจและรันบริการ
   ```powershell
   docker compose build
   # เริ่มกระบวนการ backtest ตาม config.yaml
   docker compose up backtest
   # (ทางเลือก) รันเดโมจำลองราคาออฟไลน์
   docker compose up demo
   ```
6. ผลลัพธ์จะถูกสร้างในโฟลเดอร์ `out/` (เช่น CSV, JSON, plot)
7. เมื่อใช้งานเสร็จให้หยุดด้วย `Ctrl+C` หรือใช้ `docker compose down`

### ปัญหาที่พบบ่อยบน Windows
- **Permission Denied (out/data)**: ใช้ PowerShell ที่รันด้วยสิทธิ์ปกติแต่ภายใน WSL ให้รัน `sudo chown -R $USER:$USER out data`
- **เวลาไม่ตรงโซน**: Compose ตั้งค่า TZ เป็น `Asia/Bangkok` อยู่แล้ว หากต้องการโซนอื่นให้แก้ไขใน `docker-compose.yml`

## 4. การใช้งานบน Ubuntu (Docker Engine)

1. ติดตั้ง Docker Engine และ Compose Plugin (หากยังไม่มี)
   ```bash
   sudo apt update
   sudo apt install -y ca-certificates curl gnupg
   sudo install -m 0755 -d /etc/apt/keyrings
   curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
   echo \
     "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
     "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
     sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
   sudo apt update
   sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
   sudo usermod -aG docker $USER
   newgrp docker
   ```
2. ตรวจสอบว่าสามารถรัน docker ได้ (`docker version`)
3. ภายในโฟลเดอร์โปรเจกต์ ใช้คำสั่งเดียวกับ Windows:
   ```bash
   docker compose build
   docker compose up backtest
   ```
4. สำหรับโหมดเดโม ใช้ `docker compose up demo`
5. เก็บผลลัพธ์ที่ `out/`

### ปัญหาที่พบบน Ubuntu
- หาก `docker compose` แจ้งว่าไม่รู้จักคำสั่ง ให้ลองรัน `docker --version` เพื่อตรวจสอบว่า Compose plugin ติดตั้งแล้ว
- หากไม่ได้เพิ่มผู้ใช้เข้า group `docker` ต้องรันคำสั่งด้วย `sudo`

## 5. รันโค้ดโดยตรง (ไม่ใช้ Docker) — ตัวเลือกเพิ่มเติม

> ใช้เมื่อคุณต้องการดีบักโค้ดแบบ interactive เช่นผ่าน IDE

1. สร้าง Virtual Environment
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate  # Windows PowerShell ใช้ .venv\Scripts\Activate.ps1
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
2. รัน backtest
   ```bash
   python main.py --config config.yaml --out-dir out
   ```
3. รันเดโมออฟไลน์
   ```bash
   python demo_offline.py --out-dir out/demo
   ```

## 6. โครงสร้างไฟล์สำคัญ
- `config.yaml` : ตั้งค่าคู่เทรด, ระยะเวลา, ฟีส์, features (BTD/SAH)
- `src/` : โค้ดหลักของกลยุทธ์และ backtester
- `out/` : ผลลัพธ์ที่สร้างจากการรัน (สร้างอัตโนมัติ)
- `.env` : เก็บ API key/secret (อย่า commit)

## 7. Checklist หลังติดตั้ง
- [ ] `docker compose build` สำเร็จ
- [ ] `docker compose up backtest` สร้างไฟล์ใน `out/`
- [ ] API key ถูกเซ็ตใน `.env` (หากรันโหมดที่ต้องการ)
- [ ] ปรับแต่ง `config.yaml` ตามที่ต้องการ

สิ้นสุดคู่มือ setup หากพบปัญหาเพิ่มเติมให้เช็ค `docker logs <service>` หรือตรวจสอบค่าใน `config.yaml`