@echo off
setlocal
git rev-parse --is-inside-work-tree >nul 2>&1 || goto notrepo

set REMOTE=%1
if "%REMOTE%"=="" set REMOTE=origin

set BRANCH=%2
if "%BRANCH%"=="" (
  for /f "delims=" %%b in ('git rev-parse --abbrev-ref HEAD 2^>nul') do set BRANCH=%%b
)
if "%BRANCH%"=="" set BRANCH=main

echo Syncing local branch with %REMOTE%/%BRANCH% ...
git fetch %REMOTE% || goto error
git reset --hard %REMOTE%/%BRANCH% || goto error
git clean -fd || goto error

echo Done: local tree now matches %REMOTE%/%BRANCH%.
exit /b 0

:notrepo
echo This folder is not a git repository.
exit /b 1

:error
echo Sync failed. Check the messages above for details.
exit /b 1
