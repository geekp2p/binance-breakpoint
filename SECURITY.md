
# Security Notes

- อย่าใส่ API Key/Secret ลงในไฟล์ที่ commit ได้ ตรวจใน `.gitignore` ให้เรียบร้อย
- ใช้ `.env` และ environment variables แทน
- สำหรับการใช้คีย์เทรด ควร:
  - ปิดสิทธิ์ที่ไม่จำเป็น (เช่น Futures หากไม่ใช้)
  - ใช้ **IP whitelist**
  - จำกัดสิทธิ์โอน/ถอนให้เหมาะสม
- โปรเจกต์นี้ดึง klines จาก public REST; ไม่ต้องใช้คีย์เพื่อ backtest
