# ตรวจสอบพฤติกรรมตาม `config.yaml`

เอกสารถูกเขียนเพื่อตอบคำถามว่า config ปัจจุบันให้บอททำงานตามที่อธิบายไว้หรือไม่ โดยอ้างอิงค่าจริงใน `config.yaml` เฉพาะคู่ ZECUSDT/DCRUSDT ที่ใช้พารามิเตอร์เหมือนกัน

## สรุปค่าเซตหลัก
- สัญลักษณ์ ZECUSDT และ DCRUSDT ใช้กรอบเวลา 1m ดูย้อนหลัง 2 วัน และกำหนดค่าธรรมเนียม taker/maker 0.1%.
- บันไดซื้อหลักตั้ง `d_buy` 1.5%, `m_buy` 1.5, `n_steps` 10, ระยะและขนาดเป็น Fibonacci (`spacing_mode: fibo`, `size_mode: fibonacci`) พร้อมเพดาน `max_step_drop` 25% และตัวคูณระยะ `gap_factor` 1.05.
- เปิดใช้ maker, เปิดขายแบ่งชิ้น 2 ส่วนด้วย `sell_scale_out` และไม่ได้ล็อกเพดานทุนต่อไม้/รวม (`max_quote_per_leg`/`max_total_quote` เป็น 0).

## จุดที่ตรงกับคำอธิบายในคำถาม
- **Ladder Fibonacci + จำกัดความลึก**: ค่า `spacing_mode: fibo`, `size_mode: fibonacci`, `gap_mode: multiplicative`, และ `gap_factor: 1.05` ทำให้ระยะและขนาดไล่ตามสัดส่วน Fibonacci ขยายด้วยตัวคูณ 1.05 ขณะที่ `max_step_drop: 0.25` บังคับให้ขั้นสุดท้ายไม่ลึกกว่า -25%.
- **ทุนรวมถูกครอบด้วย `b_alloc`**: แต่ละคู่กำหนด `b_alloc` (ZECUSDT 23,456 USDT, DCRUSDT 12,345 USDT) และบันได 10 ขั้นจะกระจายทุนตามลำดับ Fibonacci ภายใต้เพดานนี้.
- **Scalp/Micro แยกจากบันไดหลัก**: `scalp_mode` และ `micro_oscillation` เปิดใช้งานพร้อมสัดส่วน 33% และ 15% ตามลำดับ จึงทำคำสั่งย่อยไม่รวมกับ ladder หลัก.
- **Buy the Dip แยกและรีเบส ladder**: ตั้ง `isolate_from_ladder: true`, `rebase_ladder_from_dip: true`, และใช้ 50% ของทุนคงเหลือ (`order_pct_remaining: 0.5`) ทำให้คำสั่งดักหลุดฐานไม่ไปกินขั้นบันไดหลักและรีเซ็ตจุดอ้างอิง.
- **Sell at Height แยกจาก Profit Trail**: เปิด `sell_at_height` ด้วย threshold 4% และ pullback 0.4–2% เพื่อขาย 50% ของสถานะทันที แม้ profit trail ยังไม่ทำงาน, และตั้ง `cancel_on_miss: true`.
- **เวลาจบดีล**: `time_caps` กำหนด idle 45 นาทีให้ปิดที่กำไรขั้นต่ำ 0.4% หรือปิดทั้งหมดเมื่อครบ 180 นาทีตามที่อธิบาย.

## ประเด็นที่ควรสังเกตเพิ่มเติม
- ตัวอย่างตัวเลขไม้ที่ลึกกว่า -25% ในคำถาม (เช่นไม้ที่ 6 ที่ -26.4%) จะถูกตัดทิ้งจริง เพราะ `max_step_drop` จำกัดให้หยุดลึกสุดที่ -25%.
- ไม่มีการตั้ง `base_order_quote`/`max_quote_per_leg`/`max_total_quote` ทำให้การแบ่งทุนต่อไม้ขึ้นกับการคำนวณ Fibonacci ภายใต้ `b_alloc` ทั้งหมด หากต้องการเพดานต่อไม้ควรตั้งค่าเพิ่มเติม.
- `sell_scale_out` อยู่ใต้ `global` (ส่วนท้ายไฟล์) และตั้ง `chunks: 2`, `delay_seconds: 1`, `profit_only: true` จึงขายแบ่งสองครั้งเฉพาะเมื่อได้กำไร.

โดยรวม ค่าทั้งหมดใน `config.yaml` สอดคล้องกับคำอธิบายที่ให้มา ทั้งฝั่งซื้อ–สะสม (ladder + dip + scalp/micro) และฝั่งขาย (profit trail + sell_at_height + scale-out) พร้อมเพดานความลึกและเพดานเวลาที่ชัดเจน.