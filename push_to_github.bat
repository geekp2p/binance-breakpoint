@echo off
setlocal enabledelayedexpansion
git rev-parse --is-inside-work-tree >nul 2>&1 || goto notrepo

set REMOTE=%1
if "%REMOTE%"=="" set REMOTE=origin

set BRANCH=%2
if "%BRANCH%"=="" (
  for /f "delims=" %%b in ('git rev-parse --abbrev-ref HEAD 2^>nul') do set BRANCH=%%b
)
if "%BRANCH%"=="" set BRANCH=main

if /I "%BRANCH%"=="HEAD" (
  echo Current HEAD is detached. Defaulting branch to main.
  set BRANCH=main
)

echo Target remote: %REMOTE%
echo Target branch: %BRANCH%

git show-ref --verify --quiet refs/heads/%BRANCH%
if %ERRORLEVEL% NEQ 0 (
  echo Local branch %BRANCH% does not exist. Checkout the correct branch or specify one explicitly.
  exit /b 1
)

git remote get-url %REMOTE% >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
  echo Remote %REMOTE% is not configured. Add it with:
  echo   git remote add %REMOTE% https://github.com/your/repo.git
  exit /b 1
)

echo Pushing local %BRANCH% to %REMOTE% (force-with-lease)...
git fetch --quiet %REMOTE% %BRANCH% 2>nul
if %ERRORLEVEL% NEQ 0 (
  echo Could not pre-fetch %REMOTE%/%BRANCH%. Proceeding with push.
)

set DIRTY=
for /f "delims=" %%s in ('git status --porcelain') do set DIRTY=1
if defined DIRTY (
  echo Working tree has uncommitted or untracked changes. Commit or stash them before pushing.
  git status --short || echo (git status unavailable)
  exit /b 1
)

git rev-parse --verify %REMOTE%/%BRANCH% >nul 2>&1
if %ERRORLEVEL% EQU 0 (
  set AHEAD=0
  for /f "delims=" %%c in ('git rev-list --count %REMOTE%/%BRANCH%..%BRANCH% 2^>nul') do set AHEAD=%%c
  if "%AHEAD%"=="0" (
    echo No new commits to push. Remote %REMOTE%/%BRANCH% is already up-to-date.
    exit /b 0
  )
)

echo Sending commits (ahead by %AHEAD% of %REMOTE%/%BRANCH%)...
git push --force-with-lease %REMOTE% %BRANCH% || goto error
echo Done: remote %REMOTE%/%BRANCH% now reflects local history.
exit /b 0

:notrepo
echo This folder is not a git repository.
exit /b 1

:error
echo Push failed. Resolve issues above and retry.
exit /b 1
