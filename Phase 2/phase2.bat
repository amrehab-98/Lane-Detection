@echo off

IF "%1" == "" (
    echo "error1"
)

IF "%2" == "" (
    SET var1="%~dp0output.mp4"
    SET var2= "0"

    python phase2.py %1 %var1% %var2%
)

IF "%2" == "--debug" (
    SET var1="%~dp0output.mp4"
    SET var2= %3

    python phase2.py %1 %var1% %var2%
)

IF NOT "%2" == "" (
    IF "%3" == "" (
        SET var2= "0"
        python phase2.py %1 %2 %var2%
    )
    IF "%3" == "--debug" (
        IF NOT "%4" == "" (
            python phase2.py %1 %2 %4
        )
    )
)
pause