
# DEVELOPMENT_GUIDE — binance-breakpoint

## หลักการใหญ่ (คงเดิม)
1) งบต่อคู่ (`b_alloc`) ชัดเจน — มาติงเกลเฉพาะงบนี้เท่านั้น
2) บันไดซื้อ: ระยะทาง `d_buy`, ตัวคูณยอด `m_buy`, จำนวน `n_steps`
3) เข้า follow ที่ `P_BE*(1+p_min)`; ไต่สเตจด้วย `s1*m_step^(k-1)`
4) จุดขายตามกำไร (Floor): `F = max(H*(1 - tau*s_k), P_BE*(1+p_lock_min))`, มี **no-loss epsilon** บังคับ
5) Time-martingale: ราคานิ่ง → เพิ่ม `p_lock`, ลด `tau`, ขยายหน้าต่างเวลา
6) เพดานเวลา: idle/total-cap พร้อมกำไรขั้นต่ำเหนือ BE

## Buy the Dip (BTD) — *Scaffold (ปิดไว้)*
**เป้าหมาย**: ถ้าราคาหลุดบันไดลงไปไกล ให้ “รอให้เกิด dip แล้ว rebound” ก่อนค่อย rebase บันได เพื่อ **ไม่ซื้อแพง** และได้ฐานใหม่ต่ำกว่า  
**หลักการออกแบบ**
- Overshoot เงื่อนไข: `Low <= last_ladder_price * (1 - dip_threshold)` → `BTD_ARMED`
- รอ rebound: ราคาเด้งจาก bottom **ระหว่าง** `[rebound_min , rebound_max]`
- เมื่อคอนเฟิร์ม → *ถามผู้ใช้ก่อน* ว่าจะ:
  - `rebase_ladder_from_dip = true` → ตั้ง `P0 = bottom*(1+rebase_offset)` แล้วคำนวณบันไดใหม่
  - หรือแทรก “BTD buy” เป็นไม้พิเศษ (ถ้าตกลง)
- มี `cooldown_minutes` ป้องกันการ re-arm ถี่
> ณ เวลานี้ โค้ดเพียง **บันทึกอีเวนต์ `BTD_ARMED`** ยังไม่สั่งซื้ออัตโนมัติ

## Sell at the Height (SAH) — *Scaffold (ปิดไว้)*
**เป้าหมาย**: เมื่อราคาพุ่งสูงเกินไป ให้รอ “ย่อจากยอด/over-extended high” ก่อนขาย เพื่อเลี่ยงการขายทิ้งในเทรนด์ที่ยังแรงอยู่
- Over-extended เงื่อนไข: `High >= H*(1+height_threshold)` → `SAH_ARMED`
- รอ pullback: ย่อจากยอดอยู่ในช่วง `[pullback_min , pullback_max]` แล้วขาย (ต้อง **กำไรเสมอ** เหนือ BE)
- `cooldown_minutes` ป้องกันการ arm ถี่
> โค้ดบันทึก `SAH_ARMED` เท่านั้น ยังไม่ขายอัตโนมัติ

## การถาม-ยืนยันผู้ใช้
เมื่อเปิดใช้ฟีเจอร์ในอนาคต:
- BTD: แสดง bottom/rebound ที่ตรวจจับได้ → ขออนุมัติ “rebase ladder” หรือ “เข้าไม้ BTD”
- SAH: แสดง top/pullback ที่ตรวจจับได้ → ขออนุมัติ “ขาย at-height”
(ตอนนี้ทั้งสอง **disabled** รอคอนเฟิร์มค่าจริงก่อน)

## Roadmap
- เปิดใช้งาน BTD/SAH แบบ opt-in พร้อม unit tests
- Multi-round backtest, Portfolio mode, Metrics (DD/Expectancy/MFE/MAE), Parameter optimizer
- Paper trade (ต่อภายหลัง; ใช้ API key/secret จาก `.env` อย่างรัดกุม)
