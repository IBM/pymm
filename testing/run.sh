#!/usr/bin/bash

color_any () {
    color=$1
    shift
    echo -e "\e[${color}m$@\e[0m"
}

color_fail () {
    echo $(color_any 31 $@)
}

color_pass () {
    echo $(color_any 32 $@)
}


LOG=log.out
for i in *.py;
do
    if [ -f "$i" ] ; then
        x=`basename $i .py`
        printf "Running test: "
        printf %-20s $x
        python $i &> $LOG
        if cat $LOG | grep -q 'FAILED' ; then
            printf "$(color_fail FAILED)\n"
        elif cat $LOG | grep -q 'Segmentation fault' ; then
            printf "$(color_fail FAILED) (SEGV)\n"
        else
            printf "$(color_pass PASSED)\n"
        fi    
    fi
done
