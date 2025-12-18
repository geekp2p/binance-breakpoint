Below is a line-by-line guide to the provided config snippet. I’ve grouped related keys and explained what each field controls in plain language.

API credentials
api.key: ${BINANCE_API_KEY} – API key pulled from environment variable for authenticated Binance requests.

api.secret: ${BINANCE_API_SECRET} – API secret from environment variable for request signing.

Pair entries (repeat per symbol)
For each symbol block (e.g., ZECUSDT, DCRUSDT, DGBUSDT, BNBUSDT, TWTUSDT):

symbol – Trading pair.

quote – Quote currency (USDT).

b_alloc – Base capital allocation for this pair.

source – Market data/exchange source (Binance).

interval – Candle interval (1-minute).

lookback_days – Historical data window to load.

fees: { taker, maker } – Trading fee rates.

buy_ladder (main martingale ladder)
d_buy – Starting distance between ladder buys (e.g., 1.5% drop).

m_buy – Size multiplier for each deeper step (martingale scaling).

n_steps – Max ladder steps in the primary accumulation cycle.

spacing_mode: fibo – Step spacing follows Fibonacci pattern.

size_mode: fibonacci – Order sizing follows Fibonacci progression.

gap_mode: multiplicative – Gaps scale multiplicatively.

gap_factor – Multiplier applied to gaps (1.05 = widen slightly each step).

max_step_drop – Cap on per-step price drop fraction for ladder spacing.

base_order_quote – Fixed quote size for the first order (0 = auto/derived).

max_quote_per_leg – Max quote spent per step (0 = unlimited).

max_total_quote – Max total quote spent across the ladder (0 = unlimited).

profit_trail (take-profit trailing)
p_min – Minimum profit before trailing starts.

s1 – Initial trailing step size.

m_step – Multiplier to expand trailing step as profit grows.

tau – Trail decay/response factor.

p_lock_base – Base profit to lock once trailing activates.

p_lock_max – Max profit lock allowed.

tau_min – Minimum trail responsiveness.

no_loss_epsilon – Small buffer to avoid exiting at a tiny loss.

time_martingale (time-based reinforcement)
W1_minutes – Observation window for timing logic.

m_time – Time-based scaling multiplier.

delta_lock – Extra profit lock when time factor triggers.

beta_tau – Time-based adjustment to trailing responsiveness.

time_caps (timeouts and idle exits)
T_idle_max_minutes – Max idle time before taking action.

p_idle – Profit threshold for idle exit.

T_total_cap_minutes – Max total position duration.

p_exit_min – Minimum profit threshold for forced exit after time cap.

features
scalp_mode (fast in/out separate from main ladder)
enabled – Turn scalp trading on/off.

max_trades – Max scalp trades per cycle.

base_drop_pct, min_drop_pct, max_drop_pct – Drop triggers for scalp entries.

base_take_profit_pct, min_take_profit_pct, max_take_profit_pct – TP targets for scalps.

volatility_ref_pct – Volatility reference for scaling signals.

scale_strength – How strongly to scale orders vs. signal strength.

order_pct_allocation – Fraction of available allocation per scalp order.

cooldown_bars – Bars to wait after a scalp trade.

micro_oscillation (mean-reversion micro scalps)
enabled – Turn micro-oscillation strategy on/off.

window – Lookback window for oscillation detection.

max_band_pct – Max band width for oscillation zone.

min_swings, min_swing_pct – Minimum swing count/size to qualify.

entry_band_pct – Percentile/band used for entries.

take_profit_pct, stop_break_pct – TP and stop thresholds.

min_profit_pct – Minimum margin above break-even for exits.

volatility_stop_mult, volatility_take_profit_mult, volatility_reentry_mult – Volatility-based multipliers for stops/TP/re-entries.

order_pct_allocation – Fraction of allocation per micro order.

cooldown_bars – Bars to wait after a micro trade.

min_exit_qty – Minimum quantity for exits (0 = disabled).

buy_the_dip (dip catcher beyond ladder steps)
enabled – Turn dip-catching on/off.

dip_threshold – Drop threshold to start watching for dips.

rebound_min, rebound_max – Required rebound range to confirm dip.

cooldown_minutes – Cooldown after a dip trade.

max_orders – Max dip orders per session.

rebase_ladder_from_dip – Re-anchor ladder after a dip fill.

rebase_offset – Offset when rebasing.

order_quote – Fixed quote size per dip order (0 = use pct).

order_pct_remaining – Use % of remaining budget for dip orders.

limit_offset – Offset for limit price placement.

cancel_on_miss – Cancel if not filled.

isolate_from_ladder – Keep dip orders separate from ladder logic.

sell_at_height (rip seller on pullback)
enabled – Turn on rip-selling.

height_threshold – Overshoot threshold before watching pullback.

pullback_min, pullback_max – Required pullback range to trigger sell.

cooldown_minutes – Cooldown after a rip sell.

order_pct_position – % of position to sell on rip.

limit_offset – Limit order offset.

min_qty – Minimum quantity to place order (0 = auto).

cancel_on_miss – Cancel if not filled.

adaptive_ladder (auto-adjust ladder spacing)
enabled – Turn on adaptive spacing.

bootstrap_d_buy – Initial spacing during observation period.

bootstrap_steps – Steps to observe before adapting.

min_d_buy, max_d_buy – Bounds for adaptive spacing.

volatility_window – Lookback for volatility-based spacing calc.

sensitivity – How aggressively to adapt to volatility.

rebalance_threshold – Threshold to trigger re-spacing.

anchor_drift (structure-aware adjustments)
enabled – Turn on anchor drift logic.

structure_window – Window to detect price structure.

atr_period – ATR period for volatility.

breakout_atr_multiplier – ATR multiple to flag breakouts.

stable_band_pct – Band to consider price “stable.”

dwell_bars – Bars to wait before acting.

min_displacement_atr, min_displacement_pct – Minimum move size to treat as drift.

cooldown_bars – Bars to wait after an anchor-drift action.

plotting
plotting: { enable: true } – Enable chart/plot outputs.

general (global settings)
snapshot_every_bars – Save state every N bars.

use_maker – Prefer maker orders (limit) vs taker.

max_exit_slippage_pct – Allowed slippage for exit orders (0 disables check).

sell_scale_out: { chunks, delay_seconds, profit_only } – Scale-out parameters: number of chunks, delay between them, and whether to scale only when profitable.

binance.base_url – REST endpoint.

binance.retry – Retry attempts and backoff for API calls.

binance.save_csv / data_dir – Whether to save data and where.

bnb_fee_discount / fee_discount_factor – Use BNB fee discount or custom multiplier.

profit_recycling – Reinvest a fraction of profits.

enabled – Turn on profit recycling.

discount_allocation_pct – Portion of each round’s profit set aside to buy discounted coin.

bnb_allocation_pct – Portion of cumulative profit to buy BNB for fees.

min_order_quote – Minimum quote size for recycling orders.

discount_symbol – Target symbol for discount buys (default empty = current symbol).

bnb_symbol – Symbol used for BNB buys.

accumulation_filename – Where to store profit tracking.

rip_sell_fraction – Fraction to sell on “rip” events during recycling.



API credentials (ข้อมูลยืนยันตัวตน API)

api.key: ${BINANCE_API_KEY} – คีย์ API ที่ดึงมาจากตัวแปร Environment สำหรับเรียก Binance แบบต้องยืนยันตัวตน

api.secret: ${BINANCE_API_SECRET} – Secret ของ API ที่ดึงจากตัวแปร Environment ใช้สำหรับ “เซ็นคำขอ” (request signing)

Pair entries (ตั้งค่าต่อเหรียญ/ต่อคู่)

สำหรับแต่ละบล็อกสัญลักษณ์ (เช่น ZECUSDT, DCRUSDT, DGBUSDT, BNBUSDT, TWTUSDT):

symbol – คู่เทรด

quote – สกุลเงินฝั่ง Quote (เช่น USDT)

b_alloc – เงินทุน/งบที่กันไว้ให้คู่นี้

source – แหล่งข้อมูล/ตลาดที่ใช้ (Binance)

interval – ช่วงเวลาแท่งเทียน (เช่น 1 นาที)

lookback_days – ระยะข้อมูลย้อนหลังที่จะโหลดมาใช้

fees: { taker, maker } – อัตราค่าธรรมเนียม (taker/maker)

buy_ladder (main martingale ladder) บันไดซื้อหลักแบบมาร์ติงเกล

d_buy – ระยะห่างเริ่มต้นระหว่างไม้ซื้อในบันได (เช่น ราคาลง 1.5%)

m_buy – ตัวคูณ “ขนาดไม้” ยิ่งลงลึกยิ่งเพิ่มไม้ (martingale scaling)

n_steps – จำนวนขั้นสูงสุดของบันไดซื้อในรอบสะสมหลัก

spacing_mode: fibo – ระยะห่างแต่ละขั้นใช้รูปแบบฟีโบนัชชี

size_mode: fibonacci – ขนาดออเดอร์แต่ละไม้เพิ่มตามลำดับฟีโบนัชชี

gap_mode: multiplicative – ช่องว่างระหว่างขั้น “ขยายแบบคูณ”

gap_factor – ตัวคูณที่ใช้กับช่องว่าง (เช่น 1.05 = ค่อยๆ กว้างขึ้นทีละนิด)

max_step_drop – จำกัด % การลงต่อ “หนึ่งขั้น” ไม่ให้กว้างเกิน

base_order_quote – ขนาดออเดอร์ไม้แรกเป็นมูลค่า quote แบบ fix (0 = ให้ระบบคำนวณเอง)

max_quote_per_leg – จำกัดเงิน quote สูงสุดต่อไม้ (0 = ไม่จำกัด)

max_total_quote – จำกัดเงิน quote รวมทั้งบันได (0 = ไม่จำกัด)

profit_trail (take-profit trailing) ไล่กำไรแบบเทรล

p_min – กำไรขั้นต่ำก่อนเริ่มเปิดการเทรล

s1 – ขนาดก้าวเทรลเริ่มต้น

m_step – ตัวคูณที่ทำให้ก้าวเทรล “กว้างขึ้น” เมื่อกำไรมากขึ้น

tau – ค่าความไว/การตอบสนองของเทรล (แนวหน่วง/การไล่ตาม)

p_lock_base – กำไรพื้นฐานที่จะ “ล็อกไว้” เมื่อเทรลเริ่มทำงาน

p_lock_max – เพดานกำไรที่อนุญาตให้ล็อกได้สูงสุด

tau_min – ค่าความไวขั้นต่ำของเทรล

no_loss_epsilon – บัฟเฟอร์เล็กๆ เพื่อเลี่ยงการออกที่ “ติดลบจิ๋วๆ”

time_martingale (time-based reinforcement) เพิ่มน้ำหนักตามเวลา

W1_minutes – หน้าต่างเวลาที่ใช้สังเกตเพื่อคำนวณตรรกะเวลา

m_time – ตัวคูณการเพิ่มน้ำหนักจากปัจจัยเวลา

delta_lock – เพิ่มกำไรที่ล็อกเพิ่ม เมื่อทริกเกอร์ด้านเวลาเกิดขึ้น

beta_tau – ปรับค่าความไวของเทรล (tau) ตามปัจจัยเวลา

time_caps (timeouts and idle exits) จำกัดเวลา/ออกเมื่อเงียบ

T_idle_max_minutes – เวลานิ่งสูงสุดก่อนบังคับทำอะไรสักอย่าง

p_idle – เงื่อนไขกำไรขั้นต่ำสำหรับออกแบบ “idle exit”

T_total_cap_minutes – เวลาถือสถานะรวมสูงสุด

p_exit_min – กำไรขั้นต่ำสำหรับ “บังคับออก” เมื่อชน time cap

features (ฟีเจอร์ย่อยต่างๆ)
scalp_mode (เข้าออกเร็ว แยกจากบันไดหลัก)

enabled – เปิด/ปิด scalp

max_trades – จำนวนเทรด scalp สูงสุดต่อรอบ

base_drop_pct, min_drop_pct, max_drop_pct – เงื่อนไข % ย่อตัวเพื่อเข้าไม้ scalp (ฐาน/ต่ำสุด/สูงสุด)

base_take_profit_pct, min_take_profit_pct, max_take_profit_pct – เป้า TP ของ scalp (ฐาน/ต่ำสุด/สูงสุด)

volatility_ref_pct – ค่าความผันผวนอ้างอิงเพื่อสเกลสัญญาณ

scale_strength – ความแรงในการสเกลขนาดออเดอร์ตามความแรงสัญญาณ

order_pct_allocation – สัดส่วนของงบที่ใช้ต่อออเดอร์ scalp

cooldown_bars – พักกี่แท่งหลัง scalp หนึ่งครั้ง

micro_oscillation (ไมโครสแคปแบบ mean reversion)

enabled – เปิด/ปิดกลยุทธ์ไมโคร

window – ช่วงย้อนหลังที่ใช้ตรวจ oscillation

max_band_pct – ความกว้างแบนด์สูงสุดของโซนแกว่ง

min_swings, min_swing_pct – จำนวน/ขนาดการแกว่งขั้นต่ำจึงเข้าเงื่อนไข

entry_band_pct – เปอร์เซ็นไทล์/แบนด์ที่ใช้เป็นจุดเข้า

take_profit_pct, stop_break_pct – เป้า TP และจุด stop/ตัดออก

min_profit_pct – กำไรขั้นต่ำเหนือจุดคุ้มทุนก่อนยอมออก

volatility_stop_mult, volatility_take_profit_mult, volatility_reentry_mult – ตัวคูณ stop/TP/เข้าใหม่ตามความผันผวน

order_pct_allocation – สัดส่วนงบต่อออเดอร์ไมโคร

cooldown_bars – พักกี่แท่งหลังไมโครเทรด

min_exit_qty – ปริมาณขั้นต่ำสำหรับการออก (0 = ปิดการบังคับนี้)

buy_the_dip (ดักซื้อดิพนอกเหนือจากขั้นบันได)

enabled – เปิด/ปิดดักดิพ

dip_threshold – % ลงถึงระดับไหนเริ่มเฝ้าดิพ

rebound_min, rebound_max – ต้องเด้งกลับในช่วงนี้จึงถือว่าดิพยืนยัน

cooldown_minutes – คูลดาวน์หลังเทรดดิพ

max_orders – จำนวนออเดอร์ดิพสูงสุดต่อเซสชัน

rebase_ladder_from_dip – เติมดิพแล้วให้ “ย้ายจุดอ้างอิงบันได” ใหม่หรือไม่

rebase_offset – ระยะ offset ตอน rebase

order_quote – ขนาดออเดอร์ดิพแบบ fix เป็น quote (0 = ใช้แบบ %)

order_pct_remaining – ใช้ % ของงบที่เหลือสำหรับออเดอร์ดิพ

limit_offset – ระยะเลื่อนราคาสำหรับวาง limit

cancel_on_miss – ถ้าไม่โดน/ไม่ฟิล ให้ยกเลิกหรือไม่

isolate_from_ladder – แยกออเดอร์ดิพออกจากตรรกะบันไดหลัก

sell_at_height (ขายตอนวิ่งแรงแล้วรอ pullback)

enabled – เปิด/ปิดขายตอน “rip”

height_threshold – เกณฑ์วิ่งเกิน (overshoot) ก่อนเริ่มเฝ้า pullback

pullback_min, pullback_max – ต้องย่อลงช่วงนี้ถึงค่อยขาย

cooldown_minutes – คูลดาวน์หลังขาย rip

order_pct_position – % ของสถานะที่จะขายเมื่อเข้าเงื่อนไข

limit_offset – ระยะเลื่อนราคา limit

min_qty – ปริมาณขั้นต่ำในการวางออเดอร์ (0 = ให้ระบบคำนวณ)

cancel_on_miss – ไม่ฟิลให้ยกเลิกหรือไม่

adaptive_ladder (ปรับระยะบันไดอัตโนมัติ)

enabled – เปิด/ปิดการปรับ spacing อัตโนมัติ

bootstrap_d_buy – ระยะเริ่มต้นช่วงเก็บข้อมูลก่อนปรับ

bootstrap_steps – จำนวนขั้นที่ใช้สังเกตก่อนเริ่ม adapt

min_d_buy, max_d_buy – กรอบต่ำสุด/สูงสุดของ d_buy ที่ปรับได้

volatility_window – หน้าต่างย้อนหลังสำหรับคำนวณความผันผวนเพื่อปรับ spacing

sensitivity – ความไว/ความดุดันในการปรับตามความผันผวน

rebalance_threshold – เกณฑ์ที่ต้องถึงก่อนจะ “ปรับระยะใหม่”

anchor_drift (ปรับโครงสร้าง/จุดยึดตามสภาพราคา)

enabled – เปิด/ปิดตรรกะ anchor drift

structure_window – หน้าต่างตรวจโครงสร้างราคา

atr_period – คาบ ATR สำหรับวัดความผันผวน

breakout_atr_multiplier – ATR กี่เท่าถือว่าเป็น breakout

stable_band_pct – ช่วง % ที่ถือว่าราคา “นิ่ง/เสถียร”

dwell_bars – รออีกกี่แท่งก่อนลงมือ

min_displacement_atr, min_displacement_pct – ระยะเคลื่อนขั้นต่ำ (แบบ ATR หรือ %) จึงถือว่า drift จริง

cooldown_bars – พักกี่แท่งหลังทำ anchor-drift

plotting (การทำกราฟ)

plotting: { enable: true } – เปิดการสร้างกราฟ/ไฟล์ plot

general (ตั้งค่าทั่วไปทั้งระบบ)

snapshot_every_bars – บันทึกสถานะทุกๆ กี่แท่ง

use_maker – พยายามใช้ maker (limit) แทน taker (market)

max_exit_slippage_pct – สลิปเพจสูงสุดที่ยอมรับได้ตอนออก (0 = ปิดการตรวจ)

sell_scale_out: { chunks, delay_seconds, profit_only } – ตั้งค่าขายแบบทยอย: แบ่งกี่ไม้, หน่วงกี่วินาทีระหว่างไม้, และขายเฉพาะตอนกำไรหรือไม่

binance.base_url – URL ปลายทาง REST API

binance.retry – จำนวนครั้ง retry และ backoff ตอนเรียก API

binance.save_csv / data_dir – จะเซฟข้อมูลเป็น CSV ไหม และเซฟไว้ที่ไหน

bnb_fee_discount / fee_discount_factor – ใช้ส่วนลดค่าธรรมเนียมด้วย BNB หรือใช้ตัวคูณส่วนลดเอง

profit_recycling (นำกำไรบางส่วนไปหมุน/รีไซเคิล)

enabled – เปิด/ปิดการรีไซเคิลกำไร

discount_allocation_pct – กันกำไรแต่ละรอบเป็น % เพื่อไปซื้อเหรียญ “ราคาถูก/ส่วนลด”

bnb_allocation_pct – กันกำไรสะสมเป็น % เพื่อซื้อ BNB สำหรับค่าธรรมเนียม

min_order_quote – มูลค่า quote ขั้นต่ำสำหรับออเดอร์รีไซเคิล

discount_symbol – สัญลักษณ์เป้าหมายสำหรับซื้อส่วนลด (ว่าง = ใช้ symbol ปัจจุบัน)

bnb_symbol – สัญลักษณ์สำหรับซื้อ BNB

accumulation_filename – ไฟล์เก็บข้อมูลติดตามกำไร/สะสม

rip_sell_fraction – สัดส่วนที่จะขายเมื่อเกิดเหตุการณ์ “rip” ในโหมดรีไซเคิล