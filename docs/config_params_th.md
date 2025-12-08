# คำอธิบายพารามิเตอร์สำคัญใน `config.yaml`

## buy_ladder
- **d_buy**: ระยะห่างของขั้นแรก (จะนำไปคูณกับ spacing แต่ละชั้น).
- **spacing_mode**:
  - `geometric` (ค่าเริ่มต้น): ระยะห่างเท่ากันทุกขั้น `P0 * (1 - d_buy)^k`.
  - `fibo`: เว้นช่วงลึกขึ้นตามลำดับ Fibonacci (หรือ `d_multipliers` ที่กำหนดเอง) พร้อมเพดาน `max_step_drop` ต่อขั้น.
- **d_multipliers**: (ออปชัน) ลิสต์ตัวคูณสำหรับ `d_buy` รายขั้น ถ้าไม่ระบุจะใช้ Fibonacci อัตโนมัติ.
- **max_step_drop**: เพดาน % ดรอปต่อขั้น (กัน spacing ลึกเกินไป, ค่าเริ่มต้น 0.25 = 25%).
- **m_buy**: ตัวคูณขนาดคำสั่งของแต่ละขั้นเมื่อถอยลง (Martingale multiplier).
- **n_steps**: จำนวนขั้นสูงสุดของบันไดสะสมหลัก ถ้าครบแล้วจะไม่เปิดไม้เพิ่มจนกว่าจะรีเซ็ตรอบใหม่.
- **size_mode**:
  - `geometric` (ค่าเริ่มต้น): ขนาดไม้สเกลด้วย `m_buy` แบบ Martingale.
  - `fibonacci`: ใช้น้ำหนัก Fibonacci (หรือ `base_order_quote` เป็นตัวตั้ง) เพื่อกันการทบเร็วเกินไปในช่วงแรก.
- **gap_mode / gap_factor**:
  - `additive` (ค่าเริ่มต้น): ระยะห่างคงที่หรือคูณตาม `d_multipliers`/Fibonacci.
  - `multiplicative`: ระยะห่างลึกขึ้นแบบกำลัง (`d_buy * gap_factor^k`).
- **max_quote_per_leg / max_total_quote**: เพดานมูลค่าออเดอร์แต่ละขั้น และเพดานรวมของ ladder ต่อคู่ ลดความเสี่ยงทบไม้เร็ว.

## features.scalp_mode
- **max_trades**: จำนวนไม้ scalp สูงสุดต่อรอบ (ชุด “เข้าเร็ว–ออกเร็ว” แยกจากบันไดหลัก ไม่กินงบ n_steps).
- **order_pct_allocation**: สัดส่วนทุนต่อไม้สำหรับ scalp คิดจาก `b_alloc` ของสัญลักษณ์นั้น.
- ระยะห่างเข้า (`base_drop_pct`–`max_drop_pct`) และเป้ากำไร (`base_take_profit_pct`–`max_take_profit_pct`) จะสเกลตามความผันผวน (`volatility_ref_pct`, `scale_strength`).

## features.buy_the_dip
- **dip_threshold**: ราคาต่ำกว่าบันไดล่าสุดกี่ % ถึงมองว่า “overshoot” และเริ่มพิจารณาซื้อ.
- **rebound_min / rebound_max**: ต้องเด้งจากจุดต่ำสุดอย่างน้อย/ไม่เกินเท่าใดถึงจะยืนยันสัญญาณซื้อ.
- **max_orders**: จำนวนคำสั่ง buy dip แยกจากบันไดหลัก (ไม่กระทบ `n_steps`).
- **order_pct_remaining / order_quote**: กำหนดขนาดออเดอร์ด้วยสัดส่วนทุนที่เหลือ หรือมูลค่าเฉพาะกิจ.
- **isolate_from_ladder**: ซื้อ dip แบบแยก ไม่ปรับบันไดหลัก.

## features.sell_at_height
- **height_threshold**: ราคาสูงกว่ายอดล่าสุดกี่ % ถึงมองว่า over-extended.
- **pullback_min / pullback_max**: ต้องย่อจากยอดอย่างน้อย/ไม่เกินเท่าใดเพื่อยืนยันการขาย.
- **order_pct_position**: สัดส่วนปริมาณที่จะตั้งขายจากตำแหน่งที่ถืออยู่.

## features.adaptive_ladder
- **bootstrap_steps**: จำนวนขั้นสูงสุดในช่วงสังเกตการณ์เริ่มต้นก่อนปรับค่าอัตโนมัติ.
- **min_d_buy / max_d_buy**: เพดานต่ำ–สูงของระยะห่างบันไดที่ปรับตามความผันผวน.
- **volatility_window** และ **sensitivity**: หน้าต่างคำนวณความผันผวนและตัวคูณใช้ปรับ `d_buy`.
- **rebalance_threshold**: จะขยับระยะห่างบันไดเมื่อค่าที่คำนวณใหม่ต่างจากเดิมเกินเท่านี้.

## features.anchor_drift
- ใช้กรอบโครงสร้างราคา (`structure_window`) และ ATR (`atr_period`) เพื่อปรับ anchor เมื่อเกิด breakout หรือ drift.
- **stable_band_pct** และ **dwell_bars**: เงื่อนไขนิ่งเพื่อยืนยัน sideway ก่อนเลื่อน anchor.
- **min_displacement_atr / min_displacement_pct**: ระยะเลื่อนขั้นต่ำ (เชิง ATR หรือเปอร์เซ็นต์).
- **cooldown_bars**: เวลาพักก่อนพิจารณาเลื่อน anchor ครั้งถัดไป.

## time_caps และ time_martingale
- **time_martingale**: ปรับตัวคูณเวลา (`m_time`) และล็อกกำไรตามเวลา (`delta_lock`, `beta_tau`) เพื่อจำกัดการถือยาวเกินไป.
- **time_caps**: เพดานเวลารอ (`T_idle_max_minutes`), เวลารวมต่อรอบ (`T_total_cap_minutes`) และกำไรขั้นต่ำเมื่อหมดเวลา (`p_exit_min`).

## profit_trail
- กำหนดจุดเริ่ม trailing (`p_min`), ระยะ tighten (`s1`, `m_step`), และเพดานล็อกกำไร (`p_lock_base`–`p_lock_max`) พร้อมกันชนไม่ขายขาดทุน (`no_loss_epsilon`).