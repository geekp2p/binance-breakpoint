# Spot Normal+Option Risk Mode

คู่มือของ “โหมดปรับความเสี่ยง” สำหรับการเทรด Spot ที่เอา

ความผันผวนจากราคา (normal distribution)

กับ Implied Volatility จาก option บน Binance

มาคำนวณเป็น risk score แล้วใช้ปรับ “ขนาดไม้” ให้มากขึ้นหรือน้อยลง (หรือปิดโหมดทิ้ง) โดยไม่ไปยุ่งกับสัญญาณเทรดหลักของคุณเองครับ ✅

## ที่มาของแนวคิด (Background & Principles)

โหมดนี้ต่อยอดมาจากไอเดียในวิดีโอ Black–Scholes / Heat Equation ที่อธิบายว่า
การเปลี่ยนแปลงราคาสินทรัพย์ (โดยเฉพาะ **log-return**) ในช่วงสั้น ๆ สามารถมองได้เหมือน
**การกระจายแบบสุ่ม (diffusion)** ซึ่งมีลักษณะเป็น **Normal Distribution** เมื่อมองในสเกลเวลาเล็กพอ

- เส้นทางราคาจำนวนมาก (price paths) เมื่อซ้อนกัน จะกระจายรอบจุดศูนย์คล้ายระฆังคว่ำ (normal curve)
- สมการความร้อน (heat/diffusion equation) ใช้บอกว่าความน่าจะเป็นเหล่านี้กระจายตัวอย่างไร
- ในโลกไฟแนนซ์แบบ Black–Scholes ใช้ไอเดียนี้เพื่อบอกว่า "ราคาข้างหน้ามีโอกาสอยู่ตรงไหนบ้าง"

ในขณะเดียวกัน **Option** บนตลาดจริง (เช่น Binance) ก็สะท้อนว่า
**ตลาดกำลังกังวลเรื่องความผันผวนมากน้อยแค่ไหน** ผ่านค่า **Implied Volatility (IV)**
- ถ้า IV สูงมาก แปลว่าตลาดกำลังกังวลว่าจะมีการเหวี่ยงแรง / event ผิดปกติ
- ถ้า IV ต่ำ แปลว่าตลาดคาดว่าราคาอาจนิ่งกว่าประวัติ

ไอเดียของโหมดนี้คือ

> 1. ใช้ **Normal Distribution จากราคาจริง (realized volatility)** เพื่อวัดโอกาสที่แท่งถัดไปจะเหวี่ยงแรงกว่าปกติ
> 2. ใช้ **Option IV** เป็นมุมมองของตลาด ว่าคิดว่าความเสี่ยงในระยะสั้นสูงหรือต่ำกว่าประวัติ
> 3. รวมสองมุมมองนี้เป็น **คะแนนความเสี่ยง (risk score)** แล้วแปลงเป็นตัวคูณน้ำหนัก position

ดังนั้นโหมดนี้จึง **ไม่ออกสัญญาณเทรดเอง** แต่ทำหน้าที่เป็น *ชั้นปรับความเสี่ยง / Position Sizing Overlay*
สำหรับกลยุทธ์ที่มีอยู่แล้ว (เช่น Grid, Trend-follow, Mean Reversion ฯลฯ) โดยบอกว่า

> “ตอนนี้ตลาดเสี่ยงแค่ไหน → ควรเพิ่มหรือลดน้ำหนักดีล Spot ที่กำลังจะเข้า?”

ใช้กับ **Spot เท่านั้น** (หรือ Perp Leverage ต่ำมาก)  
และสามารถเปิด/ปิดโหมดนี้ได้ด้วย flag `RISK_MODE_ON`

---

## 0. สถานะโหมด

- `RISK_MODE_ON = true` → ใช้โหมดนี้ช่วยปรับขนาดไม้
- `RISK_MODE_ON = false` → ส่งคำสั่งตามกลยุทธ์หลักตามเดิม (ไม่ยุ่งเรื่องน้ำหนัก)

---

## 1. Input หลักที่โหมดนี้ต้องใช้

### 1.1 จากกลยุทธ์หลัก (Main Strategy)

- `direction` : ทิศทางดีลที่กลยุทธ์หลักต้องการ
  - `LONG` หรือ `SHORT` (ถ้าเล่น Spot อาจใช้ `BUY` / `SELL`)
- `base_size` : ขนาด position พื้นฐานที่กลยุทธ์หลักต้องการเปิด (จำนวนเหรียญหรือ USDT)
- `symbol` : เช่น `BTCUSDT`, `ETHUSDT`
- `tf` : timeframe หลักของกลยุทธ์ (เช่น 1m, 5m, 15m)
- `entry_price` (ประมาณการ): ราคาปัจจุบัน/ราคาเข้าโดยประมาณ

> ถ้า `direction = NONE` (ไม่มีสัญญาณ) → โหมดนี้ไม่ทำอะไร

### 1.2 ข้อมูลราคาปัจจุบัน (Spot)

สำหรับ `symbol` ที่จะเทรด:

- ลำดับราคา/แท่งเทียนล่าสุด `N` แท่ง (เช่น 100–200 แท่ง) บน timeframe `tf`
  - เอาไปใช้คำนวณ **short-term volatility**

### 1.3 ข้อมูล Options เพื่อดู Implied Volatility

จาก Binance Options (ถ้ามี):

- เลือก **ATM Options** ของ `symbol` เดียวกัน
  - Strike ใกล้ `entry_price` ที่สุด
  - Tenor สั้น ๆ เช่น 1d หรือ 2d (เป็น proxy ว่าตลาดกลัวแค่ไหนในระยะสั้น)
- ดึงค่า:
  - `ATM_IV_call`
  - `ATM_IV_put`
  - หรือใช้ `ATM_IV = max(ATM_IV_call, ATM_IV_put)` เป็นตัวแทน

> ถ้าดึงข้อมูล Option ไม่ได้ ให้ถือว่า “ไม่ใช้ IV” แต่ยังใช้ realized volatility ตามข้อ 2 ได้อยู่

---

## 2. คำนวณ Short-Term Volatility ด้วย Normal Distribution

### 2.1 Log-return แบบ intraday

จากราคา close ของแท่งเทียนล่าสุด `N` แท่ง:

```text
r_i = ln(Price_i / Price_{i-1})
for i = 1..N
```

คำนวณ:

- `mu_local = mean(r_i)`
- `sigma_local = std(r_i)`  (standard deviation)

ตีความ:

- `sigma_local` = *ความผันผวนเฉลี่ยต่อ 1 แท่ง* บน timeframe `tf`
- สมมติว่า **return ของแท่งต่อไป** มีการแจกแจง
  - `R_next ~ Normal(mu_local, sigma_local^2)`

### 2.2 Risk Score จาก Normal Distribution

วัดโอกาสที่แท่งถัดไปจะวิ่งเกิน “ระยะปลอดภัย” ที่เรากำหนดเอง เช่น `SAFE_MOVE` (%)

ตัวอย่าง:

```text
SAFE_MOVE = 0.5%  (ระยะวิ่งที่เราคิดว่าเป็นปกติใน 1 แท่งบน tf นี้)
safe_return = ln(1 + SAFE_MOVE)
Z_safe = (safe_return - mu_local) / sigma_local
P_big_move = 1 - Φ(Z_safe)     # Φ = CDF ของ standard normal
```

- ถ้า `P_big_move` สูง → มีโอกาสที่แท่งถัดไปเหวี่ยงแรงกว่าปกติ → ความเสี่ยงสูง

สร้าง **คะแนนความเสี่ยงจากราคา** (`risk_price`):

```text
if sigma_local == 0 → risk_price = LOW
else map P_big_move เป็นระดับ 0–1
```

ตัวอย่าง mapping:

- `P_big_move < 10%` → `risk_price = LOW  (0.25)`
- `10% ≤ P_big_move < 30%` → `risk_price = MEDIUM (0.5)`
- `30% ≤ P_big_move < 60%` → `risk_price = HIGH (0.75)`
- `P_big_move ≥ 60%` → `risk_price = EXTREME (1.0)`

---

## 3. เปรียบเทียบกับ Implied Volatility จาก Options

### 3.1 คำนวณค่าฐานจากอดีต (Historical)

จาก `sigma_local` (per bar) แปลงเป็น annualized rough (ใช้สำหรับเทียบ IV เท่านั้น):

```text
bars_per_day = 24*60 / tf_minutes
sigma_daily ≈ sigma_local * sqrt(bars_per_day)
sigma_annual_hist ≈ sigma_daily * sqrt(365)
```

### 3.2 เปรียบเทียบกับ ATM IV

ถ้ามี `ATM_IV`:

```text
IV_ratio = ATM_IV / sigma_annual_hist
```

สร้าง **คะแนนความเสี่ยงจาก IV** (`risk_iv`):

ตัวอย่าง mapping:

- `0.7 ≤ IV_ratio ≤ 1.3` → `risk_iv = NORMAL (0.5)`
- `1.3 < IV_ratio ≤ 2.0` → `risk_iv = HIGH (0.75)`
- `IV_ratio > 2.0` → `risk_iv = EXTREME (1.0)`
- `IV_ratio < 0.7` → `risk_iv = LOW (0.25)`

ถ้าดึง IV ไม่ได้:

- ตั้ง `risk_iv = UNKNOWN`  
  - สามารถ map เป็น `0.5` (neutral) หรือเลือกจะไม่ใช้ก็ได้

---

## 4. รวมความเสี่ยง และแปลงเป็น “ตัวคูณน้ำหนัก”

### 4.1 รวม Risk Score

ให้:

```text
risk_score = w_price * risk_price + w_iv * risk_iv
```

โดย `w_price + w_iv = 1` (เช่น `w_price = 0.6`, `w_iv = 0.4`)

### 4.2 แปลง Risk Score → Risk Level

ตัวอย่าง mapping:

```text
if risk_score < 0.35   → RISK_LEVEL = LOW
elif risk_score < 0.65 → RISK_LEVEL = NORMAL
elif risk_score < 0.85 → RISK_LEVEL = HIGH
else                   → RISK_LEVEL = EXTREME
```

### 4.3 Risk Level → Weight Multiplier

ตารางตัวอย่าง (ปรับได้ตาม style การเทรด):

| RISK_LEVEL | weight_mult (ตัวคูณน้ำหนัก) | ความหมายคร่าว ๆ                                  |
|-----------:|------------------------------|--------------------------------------------------|
| LOW        | 1.25 – 1.50                  | ตลาดนิ่ง/ความเสี่ยงต่ำ → อนุญาตให้เพิ่มขนาดไม้ |
| NORMAL     | 1.0                          | สถานการณ์ปกติ → ใช้ `base_size` ตามเดิม        |
| HIGH       | 0.5                          | เสี่ยงสูง → ลดขนาดไม้เหลือครึ่งหนึ่ง            |
| EXTREME    | 0.0 – 0.25                   | เสี่ยงมาก/ใกล้ event ใหญ่ → ไม่เข้า หรือเข้าน้อย|

สามารถนิยามเพิ่มเช่น:

- `min_size_mult` และ `max_size_mult` เพื่อกันไม่ให้ใหญ่/เล็กเกินไป

---

## 5. Logic การตัดสินใจเมื่อ RISK_MODE_ON = true

Pseudo-code:

```text
if RISK_MODE_ON == false:
    final_size = base_size
    comment = "Risk mode OFF – use base_size"
else:
    # 1) คำนวณ risk_price จาก normal distribution บนราคาปัจจุบัน
    # 2) คำนวณ risk_iv จาก ATM_IV (ถ้ามี)
    # 3) รวมเป็น risk_score และ RISK_LEVEL
    # 4) หาตัวคูณน้ำหนัก weight_mult

    final_size = base_size * weight_mult

    # Failsafe: ถ้า final_size ต่ำกว่า min_notional ไม่คุ้มค่าธรรมเนียม
    # → อาจเปลี่ยนเป็น "ไม่เข้าเทรด" แทน

    comment = f"Risk mode ON – level={RISK_LEVEL}, mult={weight_mult:.2f}"
```

เงื่อนไขพิเศษ:

- ถ้า `RISK_LEVEL = EXTREME` และ `direction` สวนกับเทรนด์แรงมาก
  - สามารถ override เป็น `final_size = 0` (skip trade)  
- สามารถล็อก `max_exposure` ต่อ symbol หรือทั้งหมด เพื่อป้องกัน over-leverage

---

## 6. Output ที่ระบบควรส่งออก

ทุกครั้งที่กลยุทธ์หลักมีสัญญาณ:

- `symbol`
- `direction`
- `base_size`
- `final_size` (หลังปรับด้วย risk mode)
- `RISK_MODE_ON` (true/false)
- `RISK_LEVEL` (LOW / NORMAL / HIGH / EXTREME)
- `risk_score`, `risk_price`, `risk_iv`
- `IV_ratio` (ถ้ามี)
- `comment` สั้น ๆ อธิบายการตัดสินใจ

ตัวอย่าง JSON Output (เพื่อ log หรือแสดงบน UI):

```json
{
  "symbol": "BTCUSDT",
  "direction": "LONG",
  "base_size": 0.10,
  "final_size": 0.05,
  "risk_mode_on": true,
  "risk_level": "HIGH",
  "risk_score": 0.72,
  "risk_price": 0.75,
  "risk_iv": 0.70,
  "iv_ratio": 1.5,
  "comment": "High risk: elevated short-term vol & IV, size halved."
}
```

---

## 7. การใช้งานเชิงแนวคิด

- โหมดนี้ทำหน้าที่เหมือน **เบรก/คันเร่งอัตโนมัติ** สำหรับกลยุทธ์หลัก
- ไม่ออกสัญญาณเข้าออกเอง แต่
  - เพิ่มน้ำหนักเมื่อความเสี่ยงต่ำ (low volatility + IV ปกติ)
  - ลดน้ำหนักหรือหยุดเทรดเมื่อความเสี่ยงสูง (ราคาเหวี่ยงแรง + IV พุ่ง)
- การใช้ Normal Distribution และ Option IV ช่วยให้การ “รู้สึกว่าตลาดเสี่ยง”  
  กลายเป็นตัวเลขที่ตรวจสอบและปรับจูนได้

เมื่อไม่ต้องการใช้แนวคิดนี้ ก็เพียงตั้ง `RISK_MODE_ON = false`  
ระบบจะกลับไปเทรดตามกลยุทธ์หลักด้วยน้ำหนักเดิมเหมือนก่อนเปิดโหมดนี้
