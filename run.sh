#!/bin/bash

disclaimer="
            --> Kitt<3ns Recommendation ENGINE <--

    Capabilities:

    Random Recommendations
    Top Listened Recommendations
    Top Included Recommendations
    Top Tags Recommendations

    Test Mode

    Please wait until the Engine is ready, then select your choice
"

running="
         Running...
"

end="
         Completed!
         Result writed to file correctly
         Bye
"

end_test="
         Test Mode Completed!
         Average Accuracy
"

command -v pypy >/dev/null
if [[ "$?" == "1" ]]; then
  whiptail --yesno "pypy is needed, proceed with installation?" 10 40
  if [[ "$?" == "0" ]]; then
    sudo apt install pypy
  else
    exit 0
  fi
fi

mem=$(free | awk 'FNR == 2 {print int($7/(1024*1024*2))}')

if [[ "$mem" == "0" ]]; then
  whiptail --msgbox "Sorry, the system does not have enough memory available (2Gb minimum is required)" 10 50
  exit 0
fi

cpu=$(nproc --all)

core=$(($mem<$cpu?$mem:$cpu))

whiptail --msgbox "$disclaimer" 20 70

mode=$(whiptail --menu "Select Operational Mode" 20 70 5 \
0 "Test Mode" \
1 "Recommendation Mode" \
2 "Script Mode" 3>&2 2>&1 1>&3)

recommendations=$(whiptail --menu "Select Recommendations Method" 20 70 10 \
0 "Random" \
1 "Top Listened" \
2 "Top Included" \
3 "Top Tags" \
4 "TF-IDF based" \
5 "Top-Tag combined TfIdf" \
6 "TfIdf combined Top-Tag" 3>&2 2>&1 1>&3)


case $mode in
    2)
        script=$(whiptail --menu "Which script do you want to run?" 20 70 5 \
        0 "Compute test set (x3)" 3>&2 2>&1 1>&3)
        /usr/bin/pypy script.py
    ;;
    1)
        /usr/bin/pypy kittens.py "$mode" "$recommendations" "$core" | whiptail --gauge "$running" 15 60 0
    ;;
    0)
        for istance in {1..3}
        do /usr/bin/pypy kittens.py "$mode" "$recommendations" "$core" "$istance" | whiptail --gauge "istance"$istance 15 60 0
        done
    ;;
esac
case $mode in
    0)
        test_results=$(cat data/test_result* | awk '{if(min==""){min=max=$1}; if($1>max) {max=$1}; if($1<min) {min=$1}; total+=$1; count+=1} END {print "avg",total/count,"max",max,"min",min}')
        whiptail --msgbox "Test Mode Completed! ""$test_results" 10 60
    ;;
    *)
        whiptail --msgbox "$end" 20 50
    ;;
esac
exit 0
