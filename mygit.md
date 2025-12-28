# mygit.md — คู่มือ Git บน Windows (PowerShell) ตั้งแต่เริ่มจน Push ได้จริง (กรณีโฟลเดอร์ไม่มี .git)

เอกสารนี้สรุปขั้นตอน “ตั้งค่า Git ตั้งแต่เริ่มต้นจน push ขึ้น GitHub ได้” สำหรับเคสที่โฟลเดอร์ในเครื่อง **ไม่เคยเป็น Git repo มาก่อน** (ไม่มี `.git`) แต่ต้องการผูกกับ repo บน GitHub และใช้งานแบบปกติ (`pull / commit / push`) ได้

> ตัวอย่าง path: `C:\binancex\binance-breakpoint\binance-breakpoint`  
> ตัวอย่าง repo: `https://github.com/geekp2p/binance-breakpoint.git`  
> ใช้ Windows PowerShell

---

## 0) หลักการสำคัญก่อนเริ่ม
- **อย่า push `.env` ขึ้น GitHub** (มักมี secret) ให้ใช้ `.env.example` แทน
- ถ้า GitHub เด้ง error `GH007: Your push would publish a private email address`  
  ต้องใช้ **GitHub no-reply email** หรือไปปรับ Settings ใน GitHub

---

## 1) ตั้งค่า user.name / user.email (Global) สำหรับการ commit
ตั้งครั้งเดียวในเครื่อง (ใช้กับทุก repo):

```powershell
git config --global user.name "geekp2p"
git config --global user.email "geekp2p@gmail.com"
git config --global --list


# Tagging Workflow (Windows PowerShell) — Commit → Push → Tag → Push Tag → (Optional) Delete Tag

โฟลเดอร์ตัวอย่าง:
- `C:\binancex\binance-breakpoint\binance-breakpoint`
- remote: `https://github.com/geekp2p/binance-breakpoint.git`
- branch: `main`

> แนวคิด: **Tag ชี้ไปที่ “commit”** ดังนั้นควร `commit + push` ให้เรียบร้อยก่อน แล้วค่อย `tag` และ `push tag`

---

## 1) Add → Commit → Push (อัปโหลดไฟล์/งานขึ้น GitHub)

```powershell
cd C:\binancex\binance-breakpoint\binance-breakpoint

git add mygit.md
git commit -m "Add mygit.md"
git push

## add addons emoji
git-commit-plugin with gitmoji
Conventional Commits