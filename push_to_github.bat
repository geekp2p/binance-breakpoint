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

echo Pushing local %BRANCH% to %REMOTE% (force-with-lease)...
git push --force-with-lease %REMOTE% %BRANCH% || goto error
echo Done: remote %REMOTE%/%BRANCH% now reflects local history.
exit /b 0

:notrepo
echo This folder is not a git repository.
exit /b 1

:error
echo Push failed. Resolve issues above and retry.
exit /b 1
