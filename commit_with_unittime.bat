@echo off
setlocal

rem Ensure we are in a git repository
git rev-parse --is-inside-work-tree >nul 2>&1 || goto notrepo

rem Exit early if there is nothing to commit
for /f "delims=" %%s in ('git status --porcelain') do goto haschanges
echo No changes to commit.
exit /b 0

:haschanges
rem Generate a Unix timestamp to use as the commit message
for /f "delims=" %%t in ('powershell -NoProfile -Command "Get-Date -UFormat %%s"') do set COMMIT_MESSAGE=%%t
if "%COMMIT_MESSAGE%"=="" goto timestampfail

rem Stage all changes and create the commit
if exist .gitignore (
  git add -A || goto error
) else (
  git add . || goto error
)
git commit -m "%COMMIT_MESSAGE%" || goto error

echo Created commit with message %COMMIT_MESSAGE%.
exit /b 0

:timestampfail
echo Failed to generate commit message from Unix time.
exit /b 1

:notrepo
echo This folder is not a git repository.
exit /b 1

:error
echo Commit failed. Resolve issues above and retry.
exit /b 1
