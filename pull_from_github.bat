@echo off
setlocal

rem Default remote URL for this repository (used when no remote is configured)
set DEFAULT_REMOTE_URL=https://github.com/geekp2p/binance-breakpoint.git
git rev-parse --is-inside-work-tree >nul 2>&1 || goto notrepo

set REMOTE=%1
if "%REMOTE%"=="" set REMOTE=origin

set BRANCH=%2
if "%BRANCH%"=="" (
  for /f "delims=" %%b in ('git rev-parse --abbrev-ref HEAD 2^>nul') do set BRANCH=%%b
)
if "%BRANCH%"=="" set BRANCH=main

echo Syncing local branch with %REMOTE%/%BRANCH% ...
git remote get-url %REMOTE% >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
  call :add_remote %REMOTE% %3
  if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
)

git fetch %REMOTE% || goto error
git reset --hard %REMOTE%/%BRANCH% || goto error
git clean -fd || goto error

echo Done: local tree now matches %REMOTE%/%BRANCH%.
exit /b 0

:add_remote
set ADD_REMOTE_NAME=%1
set ADD_REMOTE_URL=%2

if "%ADD_REMOTE_URL%"=="" if defined GIT_REMOTE_URL set ADD_REMOTE_URL=%GIT_REMOTE_URL%
if "%ADD_REMOTE_URL%"=="" if defined DEFAULT_REMOTE_URL set ADD_REMOTE_URL=%DEFAULT_REMOTE_URL%

if "%ADD_REMOTE_URL%"=="" (
  set /p ADD_REMOTE_URL=Enter URL for remote %ADD_REMOTE_NAME% (e.g. https://github.com/your/repo.git): 
)

if "%ADD_REMOTE_URL%"=="" (
  echo Remote %ADD_REMOTE_NAME% is not configured and no URL was provided.
  exit /b 1
)

git remote add %ADD_REMOTE_NAME% %ADD_REMOTE_URL% || exit /b 1
echo Added remote %ADD_REMOTE_NAME% with %ADD_REMOTE_URL%.
git fetch --quiet %ADD_REMOTE_NAME% 2>nul
exit /b 0

:notrepo
echo This folder is not a git repository.
exit /b 1

:error
echo Sync failed. Check the messages above for details.
exit /b 1
