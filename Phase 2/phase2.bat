@echo off
setlocal ENABLEDELAYEDEXPANSION

IF "%1" == "" (
    echo "error1"
    exit 1
)

IF "%2" == "" (
    SET outPath="%~dp0output.mp4"
    SET debugMode="0"
    SET phase="0"
    python phase2.py %1 !outPath! !debugMode! !phase!
    exit 0
)

IF "%2" == "--debug" (
    SET outPath="%~dp0output.mp4"
    SET debugMode="%3"
    IF "%4" == "--phase" (
        SET phase="%5"
    ) ELSE (
        SET phase="0"
    )
    python phase2.py %1 !outPath! !debugMode! !phase!
    exit 0
)

IF "%2" == "--phase" (
    SET outPath="%~dp0output.mp4"
    SET debugMode="0"
    SET phase="%3"
    python phase2.py %1 !outPath! !debugMode! !phase!
    exit 0
)

IF "%3" == "" (
    SET outPath="%2"
    SET debugMode="0"
    SET phase="0"
    python phase2.py %1 !outPath! !debugMode! !phase!
    exit 0
)

IF "%3" == "--debug" (
    SET outPath="%2"
    SET debugMode="%4"
    IF "%5" == "--phase" (
        SET phase="%6"
    ) ELSE (
        SET phase="0"
    )
    python phase2.py %1 !outPath! !debugMode! !phase!
    exit 0
)

IF "%3" == "--phase" (
    SET outPath="%2"
    SET debugMode="0"
    SET phase=%4
    python phase2.py %1 !outPath! !debugMode! !phase!
    exit 0
)

pause